#!/usr/bin/env python3
"""
Test script for the new accelerator detection system.
"""

import sys
import torch
from accelerator_detection import get_accelerator_detector, AcceleratorType

def test_accelerator_detection():
    """Test the accelerator detection system"""
    print("=== PyTorch Accelerator Detection Test ===\n")
    
    # Get detector
    detector = get_accelerator_detector()
    
    # Show PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"ROCm Version: {torch.version.hip}")
    print()
    
    # Get all devices
    all_devices = detector.available_devices
    
    print(f"Detected {len(all_devices)} device(s):\n")
    
    for i, device in enumerate(all_devices):
        print(f"Device {i}: {device.name}")
        print(f"  Type: {device.type.value.upper()}")
        print(f"  Index: {device.index}")
        print(f"  Device String: {device.device_string}")
        print(f"  Torch Device: {device.torch_device}")
        print(f"  FP16 Support: {device.supports_fp16}")
        print(f"  BF16 Support: {device.supports_bf16}")
        
        if device.memory_total:
            total_gb = device.memory_total / (1024**3)
            print(f"  Total Memory: {total_gb:.2f} GB")
        
        if device.memory_free:
            free_gb = device.memory_free / (1024**3)
            print(f"  Free Memory: {free_gb:.2f} GB")
        
        print()
    
    # Test device selection
    print("=== Device Selection Tests ===\n")
    
    best_device = detector.get_best_device()
    print(f"Best Device: {best_device.name} ({best_device.type.value})")
    
    cpu_device = detector.get_cpu_device()
    print(f"CPU Device: {cpu_device.name}")
    
    # Test by type
    for device_type in AcceleratorType:
        devices = detector.get_devices_by_type(device_type)
        if devices:
            print(f"{device_type.value.upper()} devices: {len(devices)}")
            for device in devices:
                print(f"  - {device.name}")
    
    print("\n=== Simple Tensor Test ===\n")
    
    # Test with best device
    try:
        test_device = best_device.torch_device
        print(f"Testing tensor creation on {test_device}")
        
        # Create a simple tensor
        x = torch.tensor([1.0, 2.0, 3.0]).to(test_device)
        y = torch.tensor([4.0, 5.0, 6.0]).to(test_device)
        z = x + y
        
        print(f"Tensor computation successful: {z.cpu().tolist()}")
        
        # Test autocast if supported
        from accelerator_detection import get_autocast_device_type, is_device_type_supported_for_autocast
        
        autocast_device_type = get_autocast_device_type(test_device)
        autocast_supported = is_device_type_supported_for_autocast(test_device)
        
        print(f"Autocast device type: {autocast_device_type}")
        print(f"Autocast supported: {autocast_supported}")
        
        if autocast_supported:
            with torch.autocast(device_type=autocast_device_type, dtype=torch.float16, enabled=True):
                z_autocast = x * y
                print(f"Autocast computation successful: {z_autocast.cpu().tolist()}")
        
    except Exception as e:
        print(f"Tensor test failed: {e}")
    
    print("\n=== Test Completed ===")


if __name__ == "__main__":
    test_accelerator_detection()
