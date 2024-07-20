import torch

gpu_list = []
for i in range(torch.cuda.device_count()):
    device_name = torch.cuda.get_device_properties(i).name
    gpu_list.append(device_name)

print(gpu_list)
