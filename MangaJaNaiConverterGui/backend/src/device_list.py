import torch
from accelerator_detection import get_accelerator_detector

# Get all available accelerator devices
detector = get_accelerator_detector()
all_devices = detector.available_devices

device_list = []
for device in all_devices:
    device_info = {
        "type": device.type.value,
        "index": device.index,
        "name": device.name,
        "device_string": device.device_string,
        "supports_fp16": device.supports_fp16,
        "supports_bf16": device.supports_bf16,
    }
    if device.memory_total:
        device_info["memory_total"] = device.memory_total
    if device.memory_free:
        device_info["memory_free"] = device.memory_free
    
    device_list.append(device_info)

print("Available accelerator devices:")
for i, device_info in enumerate(device_list):
    memory_info = ""
    if "memory_total" in device_info:
        total_gb = device_info["memory_total"] / (1024**3)
        memory_info = f" ({total_gb:.1f}GB)"
    
    print(f"  {i}: {device_info['name']} ({device_info['type'].upper()}:{device_info['index']}){memory_info}")
    print(f"     Device String: {device_info['device_string']}")
    print(f"     FP16 Support: {device_info['supports_fp16']}")
    print(f"     BF16 Support: {device_info['supports_bf16']}")

# Legacy CUDA device list for backward compatibility
gpu_list = []
for i in range(torch.cuda.device_count()):
    device_name = torch.cuda.get_device_properties(i).name
    gpu_list.append(device_name)

print(f"\nLegacy CUDA GPU list: {gpu_list}")
