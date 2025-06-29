import torch
import sys
import subprocess
import os

print("=== GPU 检查报告 ===")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 当前GPU
    current_device = torch.cuda.current_device()
    print(f"当前GPU: {current_device}")
    
else:
    print("没有可用的GPU，将使用CPU")
    print("建议: 如果您有NVIDIA GPU，请确保安装了CUDA和cuDNN")

# 检查nvidia-smi
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n=== NVIDIA-SMI 输出 ===")
        print(result.stdout)
    else:
        print("\nNVIDIA-SMI 不可用")
except FileNotFoundError:
    print("\nNVIDIA-SMI 未安装或不在PATH中")

# 检查环境变量
print(f"\n=== 环境变量 ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'None')}")

# 检查ultralytics是否能检测到GPU
try:
    from ultralytics import YOLO
    print("\n=== Ultralytics GPU 检查 ===")
    model = YOLO('yolov8s.pt')
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    print(f"YOLO模型设备: {device}")
except Exception as e:
    print(f"Ultralytics检查出错: {e}") 