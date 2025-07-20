import os

from accelerator_detection import AcceleratorType, get_accelerator_detector
from gpu import nvidia
from sanic.log import logger
from system import is_arm_mac

from api import GB, KB, MB, Dependency, add_package

# Get available accelerators
detector = get_accelerator_detector()
available_devices = detector.available_devices
gpu_devices = [d for d in available_devices if d.type != AcceleratorType.CPU]

# Build description based on available accelerators
accelerator_names = []
if any(d.type == AcceleratorType.CUDA for d in gpu_devices):
    accelerator_names.append("NVIDIA CUDA")
if any(d.type == AcceleratorType.ROCM for d in gpu_devices):
    accelerator_names.append("AMD ROCm")
if any(d.type == AcceleratorType.XPU for d in gpu_devices):
    accelerator_names.append("Intel XPU")
if any(d.type == AcceleratorType.MPS for d in gpu_devices):
    accelerator_names.append("Apple MPS")

general = "PyTorch uses .pth models to upscale images."

if is_arm_mac:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    package_description = f"{general} Optimized for Apple Silicon with MPS acceleration."
    inst_hint = f"{general} It is the most widely-used upscaling architecture and supports Apple Silicon acceleration."
elif accelerator_names:
    accelerator_list = ", ".join(accelerator_names)
    package_description = f"{general} Supports hardware acceleration with: {accelerator_list}."
    inst_hint = f"{general} It is the most widely-used upscaling architecture and supports multiple accelerators including {accelerator_list}."
else:
    package_description = f"{general} Running on CPU (no hardware accelerators detected)."
    inst_hint = f"{general} It is the most widely-used upscaling architecture. No hardware accelerators were detected, so it will run on CPU (which is slow)."


def get_pytorch():
    if is_arm_mac:
        return [
            Dependency(
                display_name="PyTorch",
                pypi_name="torch",
                version="2.1.2",
                size_estimate=55.8 * MB,
                auto_update=False,
            ),
            Dependency(
                display_name="TorchVision",
                pypi_name="torchvision",
                version="0.16.2",
                size_estimate=1.3 * MB,
                auto_update=False,
            ),
        ]
    else:
        return [
            Dependency(
                display_name="PyTorch",
                pypi_name="torch",
                version="2.1.2+cu121" if nvidia.is_available else "2.1.2",
                size_estimate=2 * GB if nvidia.is_available else 140 * MB,
                extra_index_url=(
                    "https://download.pytorch.org/whl/cu121"
                    if nvidia.is_available
                    else "https://download.pytorch.org/whl/cpu"
                ),
                auto_update=False,
            ),
            Dependency(
                display_name="TorchVision",
                pypi_name="torchvision",
                version="0.16.2+cu121" if nvidia.is_available else "0.16.2",
                size_estimate=2 * MB if nvidia.is_available else 800 * KB,
                extra_index_url=(
                    "https://download.pytorch.org/whl/cu121"
                    if nvidia.is_available
                    else "https://download.pytorch.org/whl/cpu"
                ),
                auto_update=False,
            ),
        ]


package = add_package(
    __file__,
    id="chaiNNer_pytorch",
    name="PyTorch",
    description=package_description,
    dependencies=[
        *get_pytorch(),
        Dependency(
            display_name="FaceXLib",
            pypi_name="facexlib",
            version="0.3.0",
            size_estimate=59.6 * KB,
        ),
        Dependency(
            display_name="Einops",
            pypi_name="einops",
            version="0.6.1",
            size_estimate=42.2 * KB,
        ),
        Dependency(
            display_name="safetensors",
            pypi_name="safetensors",
            version="0.4.0",
            size_estimate=1 * MB,
        ),
        Dependency(
            display_name="Spandrel",
            pypi_name="spandrel",
            version="0.3.4",
            size_estimate=264 * KB,
        ),
        Dependency(
            display_name="Spandrel extra architectures",
            pypi_name="spandrel_extra_arches",
            version="0.1.1",
            size_estimate=83 * KB,
        ),
    ],
    icon="PyTorch",
    color="#DD6B20",
)

pytorch_category = package.add_category(
    name="PyTorch",
    description="Nodes for using the PyTorch Neural Network Framework with images.",
    icon="PyTorch",
    color="#DD6B20",
    install_hint=inst_hint,
)

logger.debug(f"Loaded package {package.name}")
