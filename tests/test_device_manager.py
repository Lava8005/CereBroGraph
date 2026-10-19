"""TDD Test Suite for Centralized Hardware Acceleration and CUDA Memory Management."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data, Batch

from vector_engine.device_manager import DeviceManager


def test_device_detection_and_telemetry():
    """Verify hardware manager detects CUDA when available and reports valid metrics."""
    manager = DeviceManager()
    telemetry = manager.get_telemetry()

    assert "device_type" in telemetry
    assert "device_name" in telemetry
    assert telemetry["device_type"] in ["cuda", "cpu"]

    if torch.cuda.is_available():
        assert telemetry["device_type"] == "cuda"
        assert telemetry["total_memory_mb"] > 0
        assert telemetry["allocated_memory_mb"] >= 0


def test_tensor_non_blocking_transfer():
    """Verify tensors are safely cast to active hardware with asynchronous flags."""
    manager = DeviceManager()
    device = manager.device

    tensor = torch.randn(64, 5)
    transferred = manager.to_device(tensor)

    assert transferred.device.type == device.type
    assert transferred.shape == (64, 5)


def test_pyg_batch_gpu_execution():
    """Verify a batched PyG graph compiles and executes under modern AMP contexts."""
    manager = DeviceManager()
    device = manager.device

    # Mock Graph 1 (3 nodes, 2 undirected edges)
    g1 = Data(
        x=torch.ones((3, 5), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    # Mock Graph 2 (2 nodes, 1 undirected edge)
    g2 = Data(
        x=torch.ones((2, 5), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )

    batch = Batch.from_data_list([g1, g2])
    batch = manager.to_device(batch)

    assert batch.x.device.type == device.type
    assert batch.edge_index.device.type == device.type
    assert batch.batch.device.type == device.type

    # Verify Autocast context doesn't crash on device
    with manager.autocast():
        # Dummy linear transformation matching SE-GNN dimensionality
        linear = torch.nn.Linear(5, 64).to(device)
        out = linear(batch.x)
        assert out.shape == (5, 64)
        assert out.device.type == device.type


def test_cache_flush_idempotency():
    """Ensure VRAM deallocation functions cleanly without raising errors."""
    manager = DeviceManager()
    # Should execute cleanly on both CPU and CUDA targets
    manager.empty_cache()