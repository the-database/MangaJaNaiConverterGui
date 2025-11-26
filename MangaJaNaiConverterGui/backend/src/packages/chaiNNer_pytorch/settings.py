from dataclasses import dataclass

import torch
from accelerator_detection import AcceleratorType, get_accelerator_detector
from gpu import nvidia
from sanic.log import logger
from system import is_arm_mac

from api import DropdownSetting, NodeContext, NumberSetting, ToggleSetting

from . import package

# Get all available accelerator devices
detector = get_accelerator_detector()
all_devices = detector.available_devices

# Create device options, excluding CPU for the dropdown
gpu_devices = [device for device in all_devices if device.type != AcceleratorType.CPU]

if gpu_devices:
    package.add_setting(
        DropdownSetting(
            label="Accelerator Device",
            key="accelerator_device_index",
            description=(
                "Which accelerator device to use for PyTorch. This includes NVIDIA CUDA, "
                "AMD ROCm, Intel XPU, Apple MPS, and other supported accelerators."
            ),
            options=[{
                "label": f"{device.name} ({device.type.value.upper()}:{device.index})", 
                "value": str(i)
            } for i, device in enumerate(gpu_devices)],
            default="0",
        )
    )

# Legacy GPU index setting for backward compatibility (CUDA only)
if not is_arm_mac:
    cuda_devices = detector.get_devices_by_type(AcceleratorType.CUDA)
    if cuda_devices:
        package.add_setting(
            DropdownSetting(
                label="CUDA GPU (Legacy)",
                key="gpu_index",
                description=(
                    "Which CUDA GPU to use for PyTorch. This setting is deprecated - "
                    "use 'Accelerator Device' instead for full accelerator support."
                ),
                options=[{"label": device.name, "value": str(device.index)} for device in cuda_devices],
                default="0",
            )
        )

package.add_setting(
    ToggleSetting(
        label="Use CPU Mode",
        key="use_cpu",
        description=(
            "Use CPU for PyTorch instead of accelerator devices. This is much slower "
            "and not recommended unless you have no compatible accelerator."
        ),
        default=False,
    ),
)

# Determine default FP16 setting based on available devices
should_fp16 = False
gpu_devices = [device for device in all_devices if device.type != AcceleratorType.CPU]
if gpu_devices:
    # Enable FP16 by default if any GPU supports it
    should_fp16 = any(device.supports_fp16 for device in gpu_devices)
elif nvidia.is_available:
    should_fp16 = nvidia.all_support_fp16
else:
    should_fp16 = is_arm_mac

package.add_setting(
    ToggleSetting(
        label="Use FP16 Mode",
        key="use_fp16",
        description=(
            "Runs PyTorch in half-precision (FP16) mode for reduced memory usage. "
            "Automatically falls back to bfloat16 or FP32 when FP16 is not supported. "
            "Falls back to full-precision (FP32) mode when CPU mode is selected."
        ),
        default=should_fp16,
    ),
)

package.add_setting(
    NumberSetting(
        label="Memory Budget Limit (GiB)",
        key="budget_limit",
        description="Maximum memory (VRAM if GPU, RAM if CPU) to use for PyTorch inference. 0 means no limit. Memory usage measurement is not completely accurate yet; you may need to significantly adjust this budget limit via trial-and-error if it's not having the effect you want.",
        default=0,
        min=0,
        max=1024**2,
    )
)

# Add cache wipe setting for accelerator types that support it
has_accelerator_with_cache = any(
    device.type in [AcceleratorType.CUDA, AcceleratorType.ROCM, AcceleratorType.XPU] 
    for device in all_devices
)

if has_accelerator_with_cache:
    package.add_setting(
        ToggleSetting(
            label="Force Accelerator Cache Wipe (not recommended)",
            key="force_cache_wipe",
            description="Clears PyTorch's accelerator cache after each inference. This is NOT recommended, as it interferes with how PyTorch is intended to work and can significantly slow down inference time. Only enable this if you're experiencing issues with memory allocation.",
            default=False,
        )
    )


@dataclass(frozen=True)
class PyTorchSettings:
    use_cpu: bool
    use_fp16: bool
    gpu_index: int  # Legacy CUDA index
    accelerator_device_index: int  # New unified accelerator index
    budget_limit: int
    force_cache_wipe: bool = False

    # PyTorch 2.7 does not support FP16 when using CPU
    def __post_init__(self):
        if self.use_cpu and self.use_fp16:
            object.__setattr__(self, "use_fp16", False)
            logger.info("Falling back to FP32 mode for CPU.")

    @property
    def device(self) -> torch.device:
        """Get the appropriate torch device"""
        detector = get_accelerator_detector()
        
        # CPU override
        if self.use_cpu:
            return torch.device("cpu")
        
        # Try to use the new accelerator device index first
        gpu_devices = [device for device in detector.available_devices if device.type != AcceleratorType.CPU]
        
        if gpu_devices and 0 <= self.accelerator_device_index < len(gpu_devices):
            selected_device = gpu_devices[self.accelerator_device_index]
            return selected_device.torch_device
        
        # Fallback to legacy CUDA device selection for backward compatibility
        cuda_devices = detector.get_devices_by_type(AcceleratorType.CUDA)
        if cuda_devices and 0 <= self.gpu_index < len(cuda_devices):
            return torch.device(f"cuda:{self.gpu_index}")
        
        # Fallback to best available device
        best_device = detector.get_best_device(prefer_gpu=True)
        if best_device.type != AcceleratorType.CPU:
            return best_device.torch_device
        
        # Final fallback to CPU
        return torch.device("cpu")

    @property
    def accelerator_device(self) -> 'AcceleratorDevice':
        """Get the selected accelerator device info"""
        detector = get_accelerator_detector()
        
        if self.use_cpu:
            return detector.get_cpu_device()
        
        # Try to use the new accelerator device index first
        gpu_devices = [device for device in detector.available_devices if device.type != AcceleratorType.CPU]
        
        if gpu_devices and 0 <= self.accelerator_device_index < len(gpu_devices):
            return gpu_devices[self.accelerator_device_index]
        
        # Fallback to legacy CUDA device selection
        cuda_devices = detector.get_devices_by_type(AcceleratorType.CUDA)
        if cuda_devices and 0 <= self.gpu_index < len(cuda_devices):
            return cuda_devices[self.gpu_index]
        
        # Fallback to best available device
        return detector.get_best_device(prefer_gpu=True)


def get_settings(context: NodeContext) -> PyTorchSettings:
    settings = context.settings

    return PyTorchSettings(
        use_cpu=settings.get_bool("use_cpu", False),
        use_fp16=settings.get_bool("use_fp16", False),
        gpu_index=settings.get_int("gpu_index", 0, parse_str=True),
        accelerator_device_index=settings.get_int("accelerator_device_index", 0, parse_str=True),
        budget_limit=settings.get_int("budget_limit", 0, parse_str=True),
        force_cache_wipe=settings.get_bool("force_cache_wipe", False),
    )
