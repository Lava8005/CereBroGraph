"""Verification Suite for PyTorch CUDA 12.1 Integration."""

import hashlib
from pathlib import Path
import pytest
import torch

EXPECTED_TORCH_SHA256 = (
    "4bcee18f00c43c815efad8efaa3bca584ffdc8d2cd35ef4c44c814f2739d9191"
)


def compute_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash in streaming chunks to bound memory usage."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def test_downloaded_wheel_hash_if_present():
    """Verify SHA-256 of downloaded wheel matches the official PyTorch index."""
    root = Path(__file__).resolve().parent.parent
    wheels = list(root.glob("torch-2.5.1*.whl"))

    if not wheels:
        pytest.skip("No local wheel file found on disk (already installed/cleaned).")

    target_wheel = wheels[0]
    actual_hash = compute_sha256(target_wheel)
    assert actual_hash == EXPECTED_TORCH_SHA256, (
        f"Corrupted wheel detected!\nExpected: {EXPECTED_TORCH_SHA256}\nGot: {actual_hash}"
    )


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")


@requires_cuda
def test_cuda_runtime_initialization():
    """Verify PyTorch registers at least one CUDA device."""
    assert torch.cuda.device_count() >= 1, "No GPU devices registered by PyTorch."
    assert torch.cuda.get_device_name(0)


@requires_cuda
def test_fp16_tensor_core_execution():
    """Verify half-precision computation works properly on RTX Tensor Cores."""
    device = torch.device("cuda:0")
    tensor_a = torch.randn(512, 512, device=device, dtype=torch.float16)
    tensor_b = torch.randn(512, 512, device=device, dtype=torch.float16)

    result = torch.matmul(tensor_a, tensor_b)

    assert result.device.type == "cuda"
    assert result.dtype == torch.float16
    assert not torch.isnan(result).any(), "NaN values produced during matrix multiplication."