import json
import os
import sys

sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))
from accelerator_detection import get_accelerator_detector

# Get all available accelerator devices
detector = get_accelerator_detector()
all_devices = detector.available_devices
best_device = detector.get_best_device()

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

print(json.dumps({"all_devices": device_list, "best_device": all_devices.index(best_device)}))