import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 获取 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"可用 GPU 数量: {gpu_count}")
    
    # 打印每个 GPU 的名称
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA 不可用，当前无 GPU 支持")