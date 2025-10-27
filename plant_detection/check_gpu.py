import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version used to build PyTorch: {torch.version.cuda}")
print("-" * 40)

if torch.cuda.is_available():
    print("✅ SUCCESS: PyTorch can see your GPU.")
    device_count = torch.cuda.device_count()
    print(f"   Number of GPUs found: {device_count}")
    for i in range(device_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("❌ FAILURE: PyTorch cannot see your GPU.")
    print("   This means you have the CPU-only version of PyTorch installed,")
    print("   or your NVIDIA drivers and CUDA toolkit are not configured correctly.")