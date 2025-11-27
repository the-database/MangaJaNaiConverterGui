"""
Comprehensive accelerator detection for PyTorch backend.
Supports all available PyTorch accelerators in PyTorch 2.7+
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Optional, Sequence, override

import torch
from sanic.log import logger


class AcceleratorType(Enum):
    """Supported accelerator types"""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"  # AMD GPUs using ROCm
    MPS = "mps"    # Apple Metal Performance Shaders
    XPU = "xpu"    # Intel GPUs


@dataclass(frozen=True)
class AcceleratorDevice:
    """Information about an accelerator device"""
    type: AcceleratorType
    index: int
    name: str
    memory_total: Optional[int] = None
    memory_free: Optional[int] = None
    supports_fp16: bool = False
    supports_bf16: bool = False
    device_string: str = ""

    def __post_init__(self):
        if not self.device_string:
            if self.type == AcceleratorType.CPU:
                object.__setattr__(self, "device_string", "cpu")
            else:
                object.__setattr__(self, "device_string", f"{self.type.value}:{self.index}")

    @property
    def torch_device(self) -> torch.device:
        """Get the corresponding torch.device"""
        return torch.device(self.device_string)

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, AcceleratorDevice):
            return self.device_string == other.device_string
        return NotImplemented

    @override
    def __hash__(self) -> int:
        # only needed if you ever put Device in sets / dict keys
        return hash(self.device_string)


class AcceleratorDetector:
    """Detects and manages available accelerators"""

    def __init__(self):
        self._devices: Optional[list[AcceleratorDevice]] = None

    @cached_property
    def available_devices(self) -> list[AcceleratorDevice]:
        """Get all available accelerator devices"""
        if self._devices is None:
            self._devices = self._detect_all_devices()
        return self._devices

    def _detect_all_devices(self) -> list[AcceleratorDevice]:
        """Detect all available accelerator devices"""
        devices = []

        # Always add CPU
        devices.append(AcceleratorDevice(
            type=AcceleratorType.CPU,
            index=0,
            name="CPU",
            supports_fp16=False,  # PyTorch 2.7+ doesn't support FP16 on CPU
            supports_bf16=True,   # CPU supports bfloat16
        ))

        # Detect CUDA devices
        devices.extend(self._detect_cuda_devices())

        # Detect ROCm devices (ROCm uses CUDA API)
        devices.extend(self._detect_rocm_devices())

        # Detect Apple MPS
        devices.extend(self._detect_mps_devices())

        # Detect Intel XPU
        devices.extend(self._detect_xpu_devices())

        return devices

    def _detect_cuda_devices(self) -> list[AcceleratorDevice]:
        """Detect NVIDIA CUDA devices"""
        devices = []
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        device_props = torch.cuda.get_device_properties(i)
                        memory_info = torch.cuda.mem_get_info(i)
                        
                        # Determine FP16 support based on architecture
                        supports_fp16 = self._cuda_supports_fp16(device_props, i)
                        supports_bf16 = self._cuda_supports_bf16(device_props)

                        devices.append(AcceleratorDevice(
                            type=AcceleratorType.CUDA,
                            index=i,
                            name=device_props.name,
                            memory_total=device_props.total_memory,
                            memory_free=memory_info[0],
                            supports_fp16=supports_fp16,
                            supports_bf16=supports_bf16,
                        ))
                        logger.info(f"Detected CUDA device {i}: {device_props.name}")
                    except Exception as e:
                        logger.warning(f"Failed to get info for CUDA device {i}: {e}")
        except Exception as e:
            logger.info(f"CUDA not available: {e}")
        
        return devices

    def _detect_rocm_devices(self) -> list[AcceleratorDevice]:
        """Detect AMD ROCm devices"""
        devices = []
        try:
            # ROCm devices appear as CUDA devices due to HIP/CUDA compatibility
            # Check if we're actually running on ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                # This is ROCm, re-categorize CUDA devices as ROCm
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        try:
                            device_props = torch.cuda.get_device_properties(i)
                            memory_info = torch.cuda.mem_get_info(i)
                            
                            devices.append(AcceleratorDevice(
                                type=AcceleratorType.ROCM,
                                index=i,
                                name=device_props.name,
                                memory_total=device_props.total_memory,
                                memory_free=memory_info[0],
                                supports_fp16=True,  # Most modern AMD GPUs support FP16
                                supports_bf16=True,  # Modern AMD GPUs support bfloat16
                                device_string=f"cuda:{i}",  # ROCm uses cuda device string
                            ))
                            logger.info(f"Detected ROCm device {i}: {device_props.name}")
                        except Exception as e:
                            logger.warning(f"Failed to get info for ROCm device {i}: {e}")
        except Exception as e:
            logger.debug(f"ROCm detection failed: {e}")
        
        return devices

    def _detect_mps_devices(self) -> list[AcceleratorDevice]:
        """Detect Apple Metal Performance Shaders devices"""
        devices = []
        try:
            if (hasattr(torch, 'backends') and 
                hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_built() and 
                torch.backends.mps.is_available()):
                
                devices.append(AcceleratorDevice(
                    type=AcceleratorType.MPS,
                    index=0,
                    name="Apple Metal GPU",
                    supports_fp16=True,   # MPS supports FP16
                    supports_bf16=False,  # MPS doesn't support bfloat16 yet
                    device_string="mps",
                ))
                logger.info("Detected Apple MPS device")
        except Exception as e:
            logger.debug(f"MPS detection failed: {e}")
        
        return devices

    def _detect_xpu_devices(self) -> list[AcceleratorDevice]:
        """Detect Intel XPU devices"""
        devices = []
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device_count = torch.xpu.device_count()
                for i in range(device_count):
                    try:
                        device_name = torch.xpu.get_device_name(i)
                        # Try to get memory info if available
                        memory_info = None
                        try:
                            memory_info = torch.xpu.mem_get_info(i)
                        except Exception:
                            pass

                        devices.append(AcceleratorDevice(
                            type=AcceleratorType.XPU,
                            index=i,
                            name=device_name,
                            memory_total=memory_info[1] if memory_info else None,
                            memory_free=memory_info[0] if memory_info else None,
                            supports_fp16=True,   # Intel XPU supports FP16
                            supports_bf16=True,   # Intel XPU supports bfloat16
                        ))
                        logger.info(f"Detected Intel XPU device {i}: {device_name}")
                    except Exception as e:
                        logger.warning(f"Failed to get info for XPU device {i}: {e}")
        except Exception as e:
            logger.debug(f"XPU detection failed: {e}")
        
        return devices

    def _cuda_supports_fp16(self, device_props: Any, device_index: int) -> bool:
        """Check if CUDA device supports FP16"""
        try:
            # Check compute capability
            major, minor = device_props.major, device_props.minor
            compute_capability = major * 10 + minor
            
            # FP16 is supported on:
            # - Volta (7.0+) and newer architectures
            # - Some Turing cards (RTX series, not GTX 16xx)
            if compute_capability >= 70:  # Volta and newer
                return True
            elif compute_capability >= 75:  # Turing
                # For Turing, check if it's RTX (supports FP16) or GTX 16xx (doesn't)
                return "RTX" in device_props.name
            else:
                return False
        except Exception:
            # Fallback: try to actually use FP16
            try:
                test_tensor = torch.tensor([1.0], dtype=torch.float16, device=f"cuda:{device_index}")
                return True
            except Exception:
                return False

    def _cuda_supports_bf16(self, device_props: Any) -> bool:
        """Check if CUDA device supports bfloat16"""
        try:
            # bfloat16 is supported on Ampere (8.0+) and newer
            major, minor = device_props.major, device_props.minor
            compute_capability = major * 10 + minor
            return compute_capability >= 80
        except Exception:
            return False

    def get_devices_by_type(self, accelerator_type: AcceleratorType) -> list[AcceleratorDevice]:
        """Get all devices of a specific type"""
        return [device for device in self.available_devices if device.type == accelerator_type]

    def get_best_device(self, prefer_gpu: bool = True) -> AcceleratorDevice:
        """Get the best available device"""
        if not prefer_gpu:
            return self.get_cpu_device()

        # Priority order: CUDA > XPU > MPS > ROCm > CPU
        for device_type in [AcceleratorType.CUDA, AcceleratorType.XPU, AcceleratorType.MPS, 
                           AcceleratorType.ROCM]:
            devices = self.get_devices_by_type(device_type)
            if devices:
                return devices[0]  # Return the first device of this type

        return self.get_cpu_device()

    def get_cpu_device(self) -> AcceleratorDevice:
        """Get the CPU device"""
        cpu_devices = self.get_devices_by_type(AcceleratorType.CPU)
        return cpu_devices[0] if cpu_devices else AcceleratorDevice(
            type=AcceleratorType.CPU, index=0, name="CPU"
        )

    def get_device_by_index(self, device_type: AcceleratorType, index: int) -> Optional[AcceleratorDevice]:
        """Get a specific device by type and index"""
        devices = self.get_devices_by_type(device_type)
        for device in devices:
            if device.index == index:
                return device
        return None


def get_autocast_device_type(device: torch.device) -> str:
    """Get the correct device type string for torch.autocast"""
    device_type = device.type
    
    # Map device types to autocast device types
    if device_type in ["cuda", "rocm"]:  # ROCm uses cuda device type
        return "cuda"
    elif device_type == "xpu":
        return "xpu"
    elif device_type == "mps":
        # MPS doesn't support autocast yet, use cpu
        return "cpu"
    else:
        return "cpu"


def is_device_type_supported_for_autocast(device: torch.device) -> bool:
    """Check if device type supports autocast"""
    device_type = device.type
    return device_type in ["cuda", "xpu"]


# Global detector instance
_detector: Optional[AcceleratorDetector] = None


def get_accelerator_detector() -> AcceleratorDetector:
    """Get the global accelerator detector instance"""
    global _detector
    if _detector is None:
        _detector = AcceleratorDetector()
    return _detector


def get_available_devices() -> list[AcceleratorDevice]:
    """Convenience function to get all available devices"""
    return get_accelerator_detector().available_devices


def get_best_device(prefer_gpu: bool = True) -> AcceleratorDevice:
    """Convenience function to get the best available device"""
    return get_accelerator_detector().get_best_device(prefer_gpu)
