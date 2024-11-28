from __future__ import annotations

import gc

import numpy as np
import torch
from nodes.utils.utils import get_h_w_c
from spandrel import ImageModelDescriptor

from api import Progress

from ..upscale.auto_split import Split, Tiler, auto_split
from .utils import safe_cuda_cache_empty


def _into_standard_image_form(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:
        # (H, W)
        return t
    elif len(t.shape) == 3:
        # (C, H, W) -> (H, W, C)
        return t.permute(1, 2, 0)
    elif len(t.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return t.squeeze(0).permute(1, 2, 0)
    else:
        raise ValueError("Unsupported output tensor shape")


def _into_batched_form(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:
        # (H, W) -> (1, 1, H, W)
        return t.unsqueeze(0).unsqueeze(0)
    elif len(t.shape) == 3:
        # (H, W, C) -> (1, C, H, W)
        return t.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Unsupported input tensor shape")


def _rgb_to_bgr(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 3 and t.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return t.flip(2)
    elif len(t.shape) == 3 and t.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return torch.cat((t[:, :, 2:3], t[:, :, 1:2], t[:, :, 0:1], t[:, :, 3:4]), 2)
    else:
        return t


def _into_tensor(
    img: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    img = np.ascontiguousarray(img)
    writeable = img.flags.writeable
    try:
        if not writeable and device == torch.device("cpu"):
            img = np.copy(img)
        else:
            # since we are going to copy the image to the GPU, we can skip the copy here
            try:
                img.flags.writeable = True
            except Exception:
                # Some arrays cannot be made writeable, and we need to copy them
                img = np.copy(img)
        input_tensor = (
            torch.from_numpy(img).pin_memory().to(device, dtype, non_blocking=True)
        )
        return input_tensor
    finally:
        img.flags.writeable = writeable


@torch.inference_mode()
def pytorch_auto_split(
    img: np.ndarray,
    model: ImageModelDescriptor[torch.nn.Module],
    device: torch.device,
    use_fp16: bool,
    tiler: Tiler,
    progress: Progress,
) -> np.ndarray:
    dtype = torch.float32
    if use_fp16:
        if model.supports_half:
            dtype = torch.float16
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    # print("dtype", dtype, use_fp16, flush=True)
    if model.dtype != dtype or model.device != device:
        # print("move model", flush=True)
        model = model.to(device, dtype, memory_format=torch.channels_last)

    def upscale(img: np.ndarray, _: object):
        progress.check_aborted()
        if progress.paused:
            # clear resources before pausing
            gc.collect()
            safe_cuda_cache_empty()
            progress.suspend()

        input_tensor = None
        try:
            _, _, input_channels = get_h_w_c(img)
            # convert to tensor
            input_tensor = _into_tensor(img, device, dtype)
            # expand grayscale tensor to match model input channels
            if input_channels == 1 and model.input_channels > 1:
                input_tensor = input_tensor.repeat(1, 1, model.input_channels)
            else:
                input_tensor = _rgb_to_bgr(input_tensor)
            input_tensor = _into_batched_form(input_tensor)
            input_tensor = input_tensor.to(
                memory_format=torch.channels_last
            )  # TODO refactor
            # inference
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                output_tensor = model(input_tensor)

            # convert back to numpy
            output_tensor = _into_standard_image_form(output_tensor)
            if input_channels == 1:
                output_tensor = output_tensor[:, :, 0].unsqueeze(-1)
            else:
                output_tensor = _rgb_to_bgr(output_tensor)
            # print("out dtype", output_tensor.dtype, flush=True)
            # result = output_tensor.detach().cpu().detach().float().numpy()
            result = output_tensor.detach().cpu().detach()
            if result.dtype == torch.bfloat16:
                result = result.float()
            result = result.numpy()

            return result
        except RuntimeError as e:
            # Check to see if its actually the CUDA out of memory error
            if "allocate" in str(e) or "CUDA" in str(e):
                # Collect garbage (clear VRAM)
                if input_tensor is not None:
                    try:
                        input_tensor.detach().cpu()
                    except Exception:
                        pass
                    del input_tensor
                gc.collect()
                safe_cuda_cache_empty()
                return Split()
            else:
                # Re-raise the exception if not an OOM error
                raise

    return auto_split(img, upscale, tiler)
