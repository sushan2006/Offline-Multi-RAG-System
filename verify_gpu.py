#!/usr/bin/env python3
"""
Verify that PyTorch can use your NVIDIA GPU (e.g. RTX 5050).
Run: python verify_gpu.py
"""
import sys

def main():
    print("=" * 50)
    print("GPU Verification for RAG Project")
    print("=" * 50)

    try:
        import torch
    except ImportError:
        print("\n❌ PyTorch is not installed.")
        print("   Install with CUDA first:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available:  {cuda_available}")

    if cuda_available:
        print(f"✓ Device name:     {torch.cuda.get_device_name(0)}")
        print(f"✓ Device count:    {torch.cuda.device_count()}")
        print("\n🔥 GPU is connected successfully! Your RAG embeddings will use it.")
    else:
        print("\n⚠️  GPU NOT detected. PyTorch is using CPU.")
        print("   You likely have CPU-only PyTorch. Install with CUDA:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("   (Replace cu124 with cu121 or cu126 to match your nvidia-smi CUDA version)")
        sys.exit(1)

    # Quick test: move a tensor to GPU
    try:
        x = torch.randn(100, 100).cuda()
        _ = x @ x.T  # matrix multiply on GPU
        print("\n✓ GPU compute test passed.")
    except Exception as e:
        print(f"\n❌ GPU compute test failed: {e}")
        sys.exit(1)

    print("=" * 50)

if __name__ == "__main__":
    main()
