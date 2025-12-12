import argparse
import ctypes
import io
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from queue import Queue
from multiprocessing import Queue as MPQueue, Process
from threading import Thread
from typing import Any, Literal
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
import pyvips
import rarfile
from chainner_ext import ResizeFilter, resize
from cv2.typing import MatLike
from PIL import Image, ImageCms, ImageFilter
from PIL.Image import Image as ImageType
from PIL.ImageCms import ImageCmsProfile
from rarfile import RarFile
from spandrel import ImageModelDescriptor, ModelDescriptor

sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))

import spandrel_custom
from nodes.impl.image_utils import normalize, to_uint8, to_uint16
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
)
from nodes.utils.utils import get_h_w_c
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
    upscale_image_node,
)
from progress_controller import ProgressController, ProgressToken

from api import (
    NodeContext,
    SettingsParser,
)


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self, progress: ProgressToken, settings: SettingsParser, storage_dir: Path
    ) -> None:
        super().__init__()

        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir

        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0.001)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

        # TODO: send progress event

    @property
    def settings(self) -> SettingsParser:
        """
        Returns the settings of the current node execution.
        """
        return self.__settings

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


def get_tile_size(tile_size_str: str) -> TileSize:
    if tile_size_str == "Auto (Estimate)":
        return ESTIMATE
    elif tile_size_str == "Maximum":
        return MAX_TILE_SIZE
    elif tile_size_str == "No Tiling":
        return NO_TILING
    elif tile_size_str.isdecimal():
        return TileSize(int(tile_size_str))

    return ESTIMATE


"""
lanczos downscale without color conversion, for pre-upscale
downscale and final color downscale
"""


def standard_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    h, w, _ = get_h_w_c(image)
    if (w, h) == new_size:
        return image
    
    # prevent upscaling
    if new_size[0] > w or new_size[1] > h:
        return image
    
    new_image = image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.Lanczos, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    _, _, c = get_h_w_c(image)

    if c == 1 and new_image.ndim == 3:
        new_image = np.squeeze(new_image, axis=-1)

    return new_image


"""
final downscale for grayscale images only
"""


def dotgain20_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    h, w, _ = get_h_w_c(image)
    if (w, h) == new_size:
        return image
    
    # prevent upscaling
    if new_size[0] > w or new_size[1] > h:
        return image
    
    size_ratio = h / new_size[1]
    blur_size = (1 / size_ratio - 1) / 3.5
    if blur_size >= 0.1:
        blur_size = min(blur_size, 250)

    pil_image = Image.fromarray(image, mode="L")
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))
    pil_image = ImageCms.applyTransform(pil_image, dotgain20togamma1transform, False)

    new_image = np.array(pil_image)
    new_image = new_image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.CubicCatrom, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    pil_image = Image.fromarray(new_image[:, :, 0], mode="L")
    pil_image = ImageCms.applyTransform(pil_image, gamma1todotgain20transform, False)
    return np.array(pil_image)


def image_resize(
    image: np.ndarray, new_size: tuple[int, int], is_grayscale: bool
) -> np.ndarray:
    if is_grayscale:
        return dotgain20_resize(image, new_size)

    return standard_resize(image, new_size)


def get_system_codepage() -> Any:
    return None if not is_windows else ctypes.windll.kernel32.GetConsoleOutputCP()


def enhance_contrast(image: np.ndarray) -> MatLike:
    image_p = Image.fromarray(image).convert("L")

    # Calculate the histogram
    hist = image_p.histogram()
    # print(hist)

    # Find the global maximum peak in the range 0-30 for the black level
    new_black_level = 0
    global_max_black = hist[0]

    for i in range(1, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i
        # elif hist[i] < global_max_black:
        #     break

    # Continue searching at 31 and later for the black level
    continuous_count = 0
    for i in range(31, 256):
        if hist[i] > global_max_black:
            continuous_count = 0
            global_max_black = hist[i]
            new_black_level = i
        elif hist[i] < global_max_black:
            continuous_count += 1
            if continuous_count > 1:
                break

    # Find the global maximum peak in the range 255-225 for the white level
    new_white_level = 255
    global_max_white = hist[255]

    for i in range(254, 224, -1):
        if hist[i] > global_max_white:
            global_max_white = hist[i]
            new_white_level = i
        # elif hist[i] < global_max_white:
        #     break

    # Continue searching at 224 and below for the white level
    continuous_count = 0
    for i in range(223, -1, -1):
        if hist[i] > global_max_white:
            continuous_count = 0
            global_max_white = hist[i]
            new_white_level = i
        elif hist[i] < global_max_white:
            continuous_count += 1
            if continuous_count > 1:
                break

    print(
        f"Auto adjusted levels: new black level = {new_black_level}; new white level = {new_white_level}",
        flush=True,
    )

    image_array = np.array(image_p).astype("float32")
    image_array = np.maximum(image_array - new_black_level, 0) / (
        new_white_level - new_black_level
    )
    return np.clip(image_array, 0, 1)


def _read_image(img_stream: bytes, filename: str) -> np.ndarray:
    return _read_vips(img_stream)


def _read_image_from_path(path: str) -> np.ndarray:
    return pyvips.Image.new_from_file(path, access="sequential", fail=True).icc_transform("srgb").numpy()


def _read_vips(img_stream: bytes) -> np.ndarray:
    return pyvips.Image.new_from_buffer(img_stream, "", access="sequential").icc_transform("srgb").numpy()


def cv_image_is_grayscale(image: np.ndarray, user_threshold: float) -> bool:
    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    b, g, r = cv2.split(image[:, :, :3])

    ignore_threshold = user_threshold

    # getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)  # type: ignore
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)  # type: ignore
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)  # type: ignore

    # create masks to identify pure black and pure white pixels
    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))

    # combine masks to exclude both pure black and pure white pixels
    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    # exclude pure black and pure white pixels from diff_sum and image size calculation
    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    # if the entire image is pure black or pure white, return False
    if size_without_black_and_white == 0:
        return False

    # finding ratio of diff_sum with respect to size of image without pure black and pure white pixels
    ratio = diff_sum / size_without_black_and_white

    return ratio <= user_threshold / 12


def convert_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    channels = get_h_w_c(image)[2]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return image


def get_chain_for_image(
    image: np.ndarray,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    grayscale_detection_threshold: int,
) -> tuple[dict[str, Any], bool, int, int] | tuple[None, None, int, int]:
    original_height, original_width, _ = get_h_w_c(image)

    if target_width != 0 and target_height != 0:
        target_scale = min(
            target_height / original_height, target_width / original_width
        )
    if target_height != 0:
        target_scale = target_height / original_height
    elif target_width != 0:
        target_scale = target_width / original_width

    assert target_scale is not None

    is_grayscale = cv_image_is_grayscale(image, grayscale_detection_threshold)

    for chain in chains:
        if should_chain_activate_for_image(
            original_width, original_height, is_grayscale, target_scale, chain
        ):
            print("Matched Chain:", chain, flush=True)
            return chain, is_grayscale, original_width, original_height

    return None, None, original_width, original_height


def should_chain_activate_for_image(
    original_width: int,
    original_height: int,
    is_grayscale: bool,
    target_scale: float,
    chain: dict[str, Any],
) -> bool:
    min_width, min_height = (int(x) for x in chain["MinResolution"].split("x"))
    max_width, max_height = (int(x) for x in chain["MaxResolution"].split("x"))

    # resolution tests
    if min_width != 0 and min_width > original_width:
        return False
    if min_height != 0 and min_height > original_height:
        return False
    if max_width != 0 and max_width < original_width:
        return False
    if max_height != 0 and max_height < original_height:
        return False

    # color / grayscale tests
    if is_grayscale and not chain["IsGrayscale"]:
        return False
    if not is_grayscale and not chain["IsColor"]:
        return False

    # scale tests
    if chain["MaxScaleFactor"] != 0 and target_scale > chain["MaxScaleFactor"]:
        return False
    if chain["MinScaleFactor"] != 0 and target_scale < chain["MinScaleFactor"]:
        return False

    return True


def ai_upscale_image(
    image: np.ndarray, model_tile_size: TileSize, model: ImageModelDescriptor | None
) -> np.ndarray:
    if model is not None:
        result = upscale_image_node(
            context,
            image,
            model,
            False,
            0,
            model_tile_size,
            256,
            False,
        )

        _, _, c = get_h_w_c(image)

        if c == 1 and result.ndim == 3:
            result = np.squeeze(result, axis=-1)

        return result

    return image


def postprocess_image(image: np.ndarray) -> np.ndarray:
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)


def final_target_resize(
    image: np.ndarray,
    target_scale: float,
    target_width: int,
    target_height: int,
    original_width: int,
    original_height: int,
    is_grayscale: bool,
) -> np.ndarray:
    # fit to dimensions
    if target_height != 0 and target_width != 0:
        h, w, _ = get_h_w_c(image)
        # determine whether to fit to height or width
        if target_height / original_height < target_width / original_width:
            target_width = 0
        else:
            target_height = 0

    # resize height, keep proportional width
    if target_height != 0:
        h, w, _ = get_h_w_c(image)
        if h != target_height:
            return image_resize(
                image, (round(w * target_height / h), target_height), is_grayscale
            )
    # resize width, keep proportional height
    elif target_width != 0:
        h, w, _ = get_h_w_c(image)
        if w != target_width:
            return image_resize(
                image, (target_width, round(h * target_width / w)), is_grayscale
            )
    else:
        h, w, _ = get_h_w_c(image)
        new_target_height = round(original_height * target_scale)
        if h != new_target_height:
            return image_resize(
                image,
                (round(w * new_target_height / h), new_target_height),
                is_grayscale,
            )

    return image


def save_image_zip(
    image: np.ndarray,
    file_name: str,
    output_zip: ZipFile,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
) -> None:
    print(f"save image to zip: {file_name}", flush=True)

    image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
    )

    # Convert the resized image back to bytes
    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        args["lossless"] = use_lossless_compression
    buf_img = pyvips.Image.new_from_array(image).write_to_buffer(f".{image_format}", **args)
    output_buffer = io.BytesIO(buf_img)  # type: ignore

    upscaled_image_data = output_buffer.getvalue()

    # Add the resized image to the output zip
    output_zip.writestr(file_name, upscaled_image_data)


def save_image(
    image: np.ndarray,
    output_file_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
) -> None:
    print(f"save image: {output_file_path}", flush=True)

    image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
    )

    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        args["lossless"] = use_lossless_compression
    pyvips.Image.new_from_array(image).write_to_file(output_file_path, **args)


def preprocess_worker_archive(
    upscale_queue: Queue,
    input_archive_path: str,
    output_archive_path: str,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given a zip or rar path, read images out of the archive, apply auto levels, add the image to upscale queue
    """

    if input_archive_path.endswith(ZIP_EXTENSIONS):
        with ZipFile(input_archive_path, "r") as input_zip:
            preprocess_worker_archive_file(
                upscale_queue,
                input_zip,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
                grayscale_detection_threshold,
            )
    elif input_archive_path.endswith(RAR_EXTENSIONS):
        with rarfile.RarFile(input_archive_path, "r") as input_rar:
            preprocess_worker_archive_file(
                upscale_queue,
                input_rar,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
                grayscale_detection_threshold,
            )


def preprocess_worker_archive_file(
    upscale_queue: Queue,
    input_archive: RarFile | ZipFile,
    output_archive_path: str,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given an input zip or rar archive, read images out of the archive, apply auto levels, add the image to upscale queue
    """
    os.makedirs(os.path.dirname(output_archive_path), exist_ok=True)
    namelist = input_archive.namelist()
    print(f"TOTALZIP={len(namelist)}", flush=True)
    for filename in namelist:
        decoded_filename = filename
        image_data = None
        try:
            decoded_filename = decoded_filename.encode("cp437").decode(
                f"cp{system_codepage}"
            )
        except:  # noqa: E722
            pass

        # Open the file inside the input zip
        try:
            with input_archive.open(filename) as file_in_archive:
                # Read the image data

                image_data = file_in_archive.read()

                # image_bytes = io.BytesIO(image_data)
                image = _read_image(image_data, filename)
                print("read image", filename, flush=True)
                chain, is_grayscale, original_width, original_height = (
                    get_chain_for_image(
                        image,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        grayscale_detection_threshold,
                    )
                )

                if is_grayscale:
                    image = convert_image_to_grayscale(image)

                model = None
                tile_size_str = ""
                if chain is not None:
                    resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                    resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
                    resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

                    # resize width and height, distorting image
                    if (
                        resize_height_before_upscale != 0
                        and resize_width_before_upscale != 0
                    ):
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (resize_width_before_upscale, resize_height_before_upscale),
                        )
                    # resize height, keep proportional width
                    elif resize_height_before_upscale != 0:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_height_before_upscale / h),
                                resize_height_before_upscale,
                            ),
                        )
                    # resize width, keep proportional height
                    elif resize_width_before_upscale != 0:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                resize_width_before_upscale,
                                round(h * resize_width_before_upscale / w),
                            ),
                        )
                    elif resize_factor_before_upscale != 100:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_factor_before_upscale / 100),
                                round(h * resize_factor_before_upscale / 100),
                            ),
                        )
                    
                    # ensure the resized image dimensions are correctly updated    
                    original_height, original_width, _ = get_h_w_c(image) 

                    if is_grayscale and chain["AutoAdjustLevels"]:
                        image = enhance_contrast(image)
                    else:
                        image = normalize(image)

                    model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                    if model_abs_path in loaded_models:
                        model = loaded_models[model_abs_path]

                    elif os.path.exists(model_abs_path):
                        model, _, _ = load_model_node(context, Path(model_abs_path))
                        loaded_models[model_abs_path] = model

                    tile_size_str = chain["ModelTileSize"]
                else:
                    image = normalize(image)

                # image = np.ascontiguousarray(image)
                upscale_queue.put(
                    (
                        image,
                        decoded_filename,
                        True,
                        is_grayscale,
                        original_width,
                        original_height,
                        get_tile_size(tile_size_str),
                        model,
                    )
                )
        except Exception as e:
            print(
                f"could not read as image, copying file to zip instead of upscaling: {decoded_filename}, {e}",
                flush=True,
            )
            upscale_queue.put(
                (image_data, decoded_filename, False, False, None, None, None, None)
            )
        #     pass
    upscale_queue.put(UPSCALE_SENTINEL)

    # print("preprocess_worker_archive exiting")


def preprocess_worker_folder(
    upscale_queue: Queue,
    input_folder_path: str,
    output_folder_path: str,
    output_filename: str,
    upscale_images: bool,
    upscale_archives: bool,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given a folder path, recursively iterate the folder
    """
    print(
        f"preprocess_worker_folder entering {input_folder_path} {output_folder_path} {output_filename}",
        flush=True,
    )
    for root, _dirs, files in os.walk(input_folder_path):
        for filename in files:
            # for output file, create dirs if necessary, or skip if file exists and overwrite not enabled
            input_file_base = Path(filename).stem
            filename_rel = os.path.relpath(
                os.path.join(root, filename), input_folder_path
            )
            output_filename_rel = os.path.join(
                os.path.dirname(filename_rel),
                output_filename.replace("%filename%", input_file_base),
            )
            output_file_path = Path(
                os.path.join(output_folder_path, output_filename_rel)
            )

            if filename.lower().endswith(IMAGE_EXTENSIONS):  # TODO if image
                if upscale_images:
                    output_file_path = str(
                        Path(f"{output_file_path}.{image_format}")
                    ).replace("%filename%", input_file_base)

                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue

                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    image = _read_image_from_path(os.path.join(root, filename))

                    chain, is_grayscale, original_width, original_height = (
                        get_chain_for_image(
                            image,
                            target_scale,
                            target_width,
                            target_height,
                            chains,
                            grayscale_detection_threshold,
                        )
                    )

                    if is_grayscale:
                        image = convert_image_to_grayscale(image)

                    model = None
                    tile_size_str = ""
                    if chain is not None:
                        resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                        resize_height_before_upscale = chain[
                            "ResizeHeightBeforeUpscale"
                        ]
                        resize_factor_before_upscale = chain[
                            "ResizeFactorBeforeUpscale"
                        ]

                        # resize width and height, distorting image
                        if (
                            resize_height_before_upscale != 0
                            and resize_width_before_upscale != 0
                        ):
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    resize_height_before_upscale,
                                ),
                            )
                        # resize height, keep proportional width
                        elif resize_height_before_upscale != 0:
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    round(w * resize_height_before_upscale / h),
                                    resize_height_before_upscale,
                                ),
                            )
                        # resize width, keep proportional height
                        elif resize_width_before_upscale != 0:
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    round(h * resize_width_before_upscale / w),
                                ),
                            )
                        elif resize_factor_before_upscale != 100:
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    round(w * resize_factor_before_upscale / 100),
                                    round(h * resize_factor_before_upscale / 100),
                                ),
                            )
                            
                        # ensure the resized image dimensions are correctly updated    
                        original_height, original_width, _ = get_h_w_c(image) 

                        if is_grayscale and chain["AutoAdjustLevels"]:
                            image = enhance_contrast(image)
                        else:
                            image = normalize(image)

                        model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                        if model_abs_path in loaded_models:
                            model = loaded_models[model_abs_path]

                        elif os.path.exists(model_abs_path):
                            model, _, _ = load_model_node(context, Path(model_abs_path))
                            loaded_models[model_abs_path] = model
                        tile_size_str = chain["ModelTileSize"]
                    else:
                        image = normalize(image)

                    # image = np.ascontiguousarray(image)

                    upscale_queue.put(
                        (
                            image,
                            output_filename_rel,
                            True,
                            is_grayscale,
                            original_width,
                            original_height,
                            get_tile_size(tile_size_str),
                            model,
                        )
                    )
            elif filename.lower().endswith(ARCHIVE_EXTENSIONS):
                if upscale_archives:
                    output_file_path = f"{output_file_path}.cbz"
                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    upscale_archive_file(
                        os.path.join(root, filename),
                        output_file_path,
                        image_format,
                        lossy_compression_quality,
                        use_lossless_compression,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        loaded_models,
                        grayscale_detection_threshold,
                    )  # TODO custom output extension
    upscale_queue.put(UPSCALE_SENTINEL)
    # print("preprocess_worker_folder exiting")


def preprocess_worker_image(
    upscale_queue: Queue,
    input_image_path: str,
    output_image_path: str,
    overwrite_existing_files: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given an image path, apply auto levels and add to upscale queue
    """
    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}", flush=True)
            return

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # with Image.open(input_image_path) as img:
        image = _read_image_from_path(input_image_path)

        chain, is_grayscale, original_width, original_height = get_chain_for_image(
            image,
            target_scale,
            target_width,
            target_height,
            chains,
            grayscale_detection_threshold,
        )

        if is_grayscale:
            image = convert_image_to_grayscale(image)

        model = None
        tile_size_str = ""
        if chain is not None:
            resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
            resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
            resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

            # resize width and height, distorting image
            if resize_height_before_upscale != 0 and resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image, (resize_width_before_upscale, resize_height_before_upscale)
                )
            # resize height, keep proportional width
            elif resize_height_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image,
                    (
                        round(w * resize_height_before_upscale / h),
                        resize_height_before_upscale,
                    ),
                )
            # resize width, keep proportional height
            elif resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image,
                    (
                        resize_width_before_upscale,
                        round(h * resize_width_before_upscale / w),
                    ),
                )
            elif resize_factor_before_upscale != 100:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image,
                    (
                        round(w * resize_factor_before_upscale / 100),
                        round(h * resize_factor_before_upscale / 100),
                    ),
                )
                
            # ensure the resized image dimensions are correctly updated    
            original_height, original_width, _ = get_h_w_c(image) 

            if is_grayscale and chain["AutoAdjustLevels"]:
                image = enhance_contrast(image)
            else:
                image = normalize(image)

            if chain["ModelFilePath"] == "No Model":
                pass
            else:
                model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                if not os.path.exists(model_abs_path):
                    raise FileNotFoundError(model_abs_path)

                if model_abs_path in loaded_models:
                    model = loaded_models[model_abs_path]

                elif os.path.exists(model_abs_path):
                    model, _, _ = load_model_node(context, Path(model_abs_path))
                    loaded_models[model_abs_path] = model
                tile_size_str = chain["ModelTileSize"]
        else:
            print("No chain!!!!!!!")
            image = normalize(image)

        # image = np.ascontiguousarray(image)

        upscale_queue.put(
            (
                image,
                None,
                True,
                is_grayscale,
                original_width,
                original_height,
                get_tile_size(tile_size_str),
                model,
            )
        )
    upscale_queue.put(UPSCALE_SENTINEL)


def upscale_worker(upscale_queue: Queue, postprocess_queue: Queue) -> None:
    """
    wait for upscale queue, for each queue entry, upscale image and add result to postprocess queue
    """
    # print("upscale_worker entering")
    while True:
        (
            image,
            file_name,
            is_image,
            is_grayscale,
            original_width,
            original_height,
            model_tile_size,
            model,
        ) = upscale_queue.get()
        if image is None:
            break

        if is_image:
            image = ai_upscale_image(image, model_tile_size, model)

            # convert back to grayscale
            if is_grayscale:
                image = convert_image_to_grayscale(image)

        postprocess_queue.put(
            (image, file_name, is_image, is_grayscale, original_width, original_height)
        )
    postprocess_queue.put(POSTPROCESS_SENTINEL)
    # print("upscale_worker exiting")


def postprocess_worker_zip(
    postprocess_queue: Queue,
    output_zip_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for postprocess queue, for each queue entry, save the image to the zip file
    """
    # print("postprocess_worker_zip entering")
    with ZipFile(output_zip_path, "w", ZIP_DEFLATED) as output_zip:
        while True:
            (
                image,
                file_name,
                is_image,
                is_grayscale,
                original_width,
                original_height,
            ) = postprocess_queue.get()
            if image is None:
                break
            if is_image:
                # image = postprocess_image(image)
                save_image_zip(
                    image,
                    str(Path(file_name).with_suffix(f".{image_format}")),
                    output_zip,
                    image_format,
                    lossy_compression_quality,
                    use_lossless_compression,
                    original_width,
                    original_height,
                    target_scale,
                    target_width,
                    target_height,
                    is_grayscale,
                )
            else:  # copy file
                output_zip.writestr(file_name, image)
            print("PROGRESS=postprocess_worker_zip_image", flush=True)
        print("PROGRESS=postprocess_worker_zip_archive", flush=True)


def postprocess_worker_folder(
    postprocess_queue: Queue,
    output_folder_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for postprocess queue, for each queue entry, save the image to the output folder
    """
    # print("postprocess_worker_folder entering")
    while True:
        image, file_name, _, is_grayscale, original_width, original_height = (
            postprocess_queue.get()
        )
        if image is None:
            break
        image = postprocess_image(image)
        save_image(
            image,
            os.path.join(output_folder_path, str(Path(f"{file_name}.{image_format}"))),
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
        )
        print("PROGRESS=postprocess_worker_folder", flush=True)

    # print("postprocess_worker_folder exiting")


def postprocess_worker_image(
    postprocess_queue: Queue,
    output_file_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for postprocess queue, for each queue entry, save the image to the output file path
    """
    while True:
        image, _, _, is_grayscale, original_width, original_height = (
            postprocess_queue.get()
        )
        if image is None:
            break
        # image = postprocess_image(image)

        save_image(
            image,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            original_width,
            original_height,
            target_scale,
            target_width,
            target_height,
            is_grayscale,
        )
        print("PROGRESS=postprocess_worker_image", flush=True)


def upscale_archive_file(
    input_zip_path: str,
    output_zip_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    # TODO accept multiple paths to reuse simple queues?

    upscale_queue = Queue(maxsize=1)
    postprocess_queue = MPQueue(maxsize=1)

    # start preprocess zip process
    preprocess_process = Thread(
        target=preprocess_worker_archive,
        args=(
            upscale_queue,
            input_zip_path,
            output_zip_path,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess zip process
    postprocess_process = Process(
        target=postprocess_worker_zip,
        args=(
            postprocess_queue,
            output_zip_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_image_file(
    input_image_path: str,
    output_image_path: str,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = MPQueue(maxsize=1)

    # start preprocess image process
    preprocess_process = Thread(
        target=preprocess_worker_image,
        args=(
            upscale_queue,
            input_image_path,
            output_image_path,
            overwrite_existing_files,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess image process
    postprocess_process = Process(
        target=postprocess_worker_image,
        args=(
            postprocess_queue,
            output_image_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


def upscale_file(
    input_file_path: str,
    output_folder_path: str,
    output_filename: str,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    input_file_base = Path(input_file_path).stem

    if input_file_path.lower().endswith(ARCHIVE_EXTENSIONS):
        output_file_path = str(
            Path(
                f"{os.path.join(output_folder_path,output_filename.replace('%filename%', input_file_base))}.cbz"
            )
        )
        print("output_file_path", output_file_path, flush=True)
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_archive_file(
            input_file_path,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        )

    elif input_file_path.lower().endswith(IMAGE_EXTENSIONS):
        output_file_path = str(
            Path(
                f"{os.path.join(output_folder_path,output_filename.replace('%filename%', input_file_base))}.{image_format}"
            )
        )
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_image_file(
            input_file_path,
            output_file_path,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        )


def upscale_folder(
    input_folder_path: str,
    output_folder_path: str,
    output_filename: str,
    upscale_images: bool,
    upscale_archives: bool,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    # print("upscale_folder: entering")

    # preprocess_queue = Queue(maxsize=1)
    upscale_queue = Queue(maxsize=1)
    postprocess_queue = MPQueue(maxsize=1)

    # start preprocess folder process
    preprocess_process = Thread(
        target=preprocess_worker_folder,
        args=(
            upscale_queue,
            input_folder_path,
            output_folder_path,
            output_filename,
            upscale_images,
            upscale_archives,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker, args=(upscale_queue, postprocess_queue)
    )
    upscale_process.start()

    # start postprocess folder process
    postprocess_process = Process(
        target=postprocess_worker_folder,
        args=(
            postprocess_queue,
            output_folder_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    postprocess_process.join()


current_file_directory = os.path.dirname(os.path.abspath(__file__))


def get_model_abs_path(chain_model_file_path: str) -> str:
    return os.path.abspath(os.path.join(models_directory, chain_model_file_path))


def get_gamma_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Custom Gray Gamma 1.0.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def get_dot20_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Dot Gain 20%.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def parse_settings_from_cli():
    parser = argparse.ArgumentParser(prog="python run_upscale.py",
                                     description="By default, used by MangaJaNaiConverterGui as an internal tool. "
                                                 "Alternative options made available to make it easier to skip the GUI "
                                                 "and run upscaling jobs directly from CLI.")

    execution_type_group = parser.add_mutually_exclusive_group(required=True)
    execution_type_group.add_argument("--settings",
                                      help="Default behaviour, based on provided appstate configuration. "
                                           "For advanced usage.")
    execution_type_group.add_argument("-f", "--file-path",
                                      help="Upscale single file")
    execution_type_group.add_argument("-d", "--folder-path",
                                      help="Upscale whole directory")

    parser.add_argument("-o", "--output-folder-path",
                        default=os.path.join(".", "out"),
                        help="Output directory for upscaled files. Default: ./out")
    parser.add_argument("-m", "--models-directory-path",
                        default=os.path.join("..", "models"),
                        help="Directory with models used for upscaling. "
                             "Supports only models bundled with MangaJaNaiConvertedGui. "
                             "Default: MangaJaNaiConverterGui/chaiNNer/models/")
    parser.add_argument("-u", "--upscale-factor",
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=2,
                        help="Used for calculating which model will be used. Default: 2")
    parser.add_argument("--device-index",
                        type=int,
                        default=0,
                        help="Device used to run upscaling jobs in case more than one is available. Default: 0")

    args = parser.parse_args()

    return parse_auto_settings(args) if args.settings else parse_manual_settings(args)


def parse_auto_settings(args):
    with open(args.settings, encoding="utf-8") as f:
        json_settings = json.load(f)

    return json_settings


def parse_manual_settings(args):
    default_file_path = os.path.join("..", "resources", "default_cli_configuration.json")
    with open(default_file_path, "r") as default_file:
        default_json = json.load(default_file)

    default_json["SelectedDeviceIndex"] = int(args.device_index)
    default_json["ModelsDirectory"] = args.models_directory_path

    default_json["Workflows"]["$values"][0]["OutputFolderPath"] = args.output_folder_path
    default_json["Workflows"]["$values"][0]["SelectedDeviceIndex"] = args.device_index
    default_json["Workflows"]["$values"][0]["UpscaleScaleFactor"] = args.upscale_factor
    if args.file_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 0
        default_json["Workflows"]["$values"][0]["InputFilePath"] = args.file_path
    elif args.folder_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 1
        default_json["Workflows"]["$values"][0]["InputFolderPath"] = args.folder_path

    return default_json


is_windows = platform.system() == "win32"
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

settings = parse_settings_from_cli()

workflow = settings["Workflows"]["$values"][settings["SelectedWorkflowIndex"]]
models_directory = settings["ModelsDirectory"]

UPSCALE_SENTINEL = (None, None, None, None, None, None, None, None)
POSTPROCESS_SENTINEL = (None, None, None, None, None, None)
CV2_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_EXTENSIONS = (*CV2_IMAGE_EXTENSIONS, ".avif")
ZIP_EXTENSIONS = (".zip", ".cbz")
RAR_EXTENSIONS = (".rar", ".cbr")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + RAR_EXTENSIONS
loaded_models = {}
system_codepage = get_system_codepage()

settings_parser = SettingsParser(
    {
        "use_cpu": settings["SelectedDeviceIndex"] == 0,
        "use_fp16": settings["UseFp16"],
        "accelerator_device_index": settings["SelectedDeviceIndex"],
        "budget_limit": 0,
    }
)

print("settings", settings_parser.get_int("accelerator_device_index", 0), flush=True)

context = _ExecutorNodeContext(ProgressController(), settings_parser, Path())

gamma1icc = get_gamma_icc_profile()
dotgain20icc = get_dot20_icc_profile()

dotgain20togamma1transform = ImageCms.buildTransformFromOpenProfiles(
    dotgain20icc, gamma1icc, "L", "L"
)
gamma1todotgain20transform = ImageCms.buildTransformFromOpenProfiles(
    gamma1icc, dotgain20icc, "L", "L"
)

if __name__ == "__main__":
    spandrel_custom.install()
    # gc.disable() #TODO!!!!!!!!!!!!
    # Record the start time
    start_time = time.time()

    image_format = None
    if workflow["WebpSelected"]:
        image_format = "webp"
    elif workflow["PngSelected"]:
        image_format = "png"
    elif workflow["AvifSelected"]:
        image_format = "avif"
    else:
        image_format = "jpeg"

    target_scale: float | None = None
    target_width = 0
    target_height = 0

    grayscale_detection_threshold = workflow["GrayscaleDetectionThreshold"]

    if workflow["ModeScaleSelected"]:
        target_scale = workflow["UpscaleScaleFactor"]
    elif workflow["ModeWidthSelected"]:
        target_width = workflow["ResizeWidthAfterUpscale"]
    elif workflow["ModeHeightSelected"]:
        target_height = workflow["ResizeHeightAfterUpscale"]
    else:
        target_width = workflow["DisplayDeviceWidth"]
        target_height = workflow["DisplayDeviceHeight"]

    if workflow["SelectedTabIndex"] == 1:
        upscale_folder(
            workflow["InputFolderPath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["UpscaleImages"],
            workflow["UpscaleArchives"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
            grayscale_detection_threshold,
        )
    elif workflow["SelectedTabIndex"] == 0:
        upscale_file(
            workflow["InputFilePath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
            grayscale_detection_threshold,
        )

    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
