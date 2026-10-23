"""CUDA Runtime & Hardware Acceleration Test Suite."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")


def test_torch_cuda_availability():
    """Verify PyTorch is compiled with CUDA support and GPU is accessible."""
    assert torch.cuda.is_available()


def test_gpu_hardware_identity():
    """Verify a CUDA device is detected and reports a name."""
    device_count = torch.cuda.device_count()
    assert device_count >= 1, "No CUDA devices detected."

    assert torch.cuda.get_device_name(0)


def test_cuda_tensor_allocation_and_matmul():
    """Verify GPU VRAM allocation, FP16 execution, and kernel computation."""
    device = torch.device("cuda:0")

    # Allocate matrices directly in VRAM
    matrix_a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    matrix_b = torch.randn(1024, 1024, device=device, dtype=torch.float16)

    # Compute matrix multiplication on Tensor Cores
    result = torch.matmul(matrix_a, matrix_b)

    assert result.device.type == "cuda"
    assert result.shape == (1024, 1024)
    assert not torch.isnan(result).any(), "GPU computation produced NaN values."


def test_cuda_memory_cleanup():
    """Verify VRAM deallocation to prevent Out-Of-Memory (OOM) leaks during GNN execution."""
    torch.cuda.empty_cache()
    allocated_bytes = torch.cuda.memory_allocated(0)
    # Residual allocation must remain low when no tensors are active
    assert allocated_bytes < 50 * 1024 * 1024, (
        f"VRAM leak detected: {allocated_bytes / (1024**2):.2f} MB still occupied."
    )