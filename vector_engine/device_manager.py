"""Centralized Hardware Acceleration & CUDA Memory Lifecycle Manager.

Provides deterministic device placement, asynchronous non-blocking memory transfers,
automatic mixed precision (AMP) scopes, and VRAM telemetry for CerebroGraph.
"""

import contextlib
import logging
from typing import Any, Dict, Generator, Optional, Union

import torch
from torch_geometric.data import Batch, Data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DeviceManager:
    """Manages accelerator state, device placement, and memory compaction."""

    def __init__(self, force_cpu: bool = False):
        self.device = self._resolve_device(force_cpu)
        self.is_cuda = self.device.type == "cuda"
        self._log_initialization()

    def _resolve_device(self, force_cpu: bool) -> torch.device:
        if not force_cpu and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _log_initialization(self) -> None:
        telemetry = self.get_telemetry()
        if self.is_cuda:
            logging.info(
                "GPU Engine Activated: %s | VRAM Total: %d MB | Compute Capability: %s",
                telemetry["device_name"],
                telemetry["total_memory_mb"],
                telemetry.get("compute_capability", "N/A"),
            )
        else:
            logging.warning("Hardware Acceleration Disabled: Utilizing CPU Fallback.")

    def get_telemetry(self) -> Dict[str, Any]:
        """Return runtime hardware specifications and current VRAM utilization."""
        data: Dict[str, Any] = {
            "device_type": self.device.type,
            "device_index": self.device.index if self.is_cuda else None,
            "device_name": torch.cuda.get_device_name(self.device) if self.is_cuda else "Host CPU",
        }

        if self.is_cuda:
            props = torch.cuda.get_device_properties(self.device)
            data.update({
                "total_memory_mb": round(props.total_memory / (1024**2)),
                "allocated_memory_mb": round(torch.cuda.memory_allocated(self.device) / (1024**2)),
                "reserved_memory_mb": round(torch.cuda.memory_reserved(self.device) / (1024**2)),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return data

    def to_device(
        self,
        target: Union[torch.Tensor, Data, Batch],
        non_blocking: bool = True,
    ) -> Union[torch.Tensor, Data, Batch]:
        """Cast tensors or PyG graphs to the target device asynchronously."""
        if hasattr(target, "to"):
            if isinstance(target, (Data, Batch)):
                return target.to(self.device)
            return target.to(self.device, non_blocking=non_blocking if self.is_cuda else False)
        return target

    @contextlib.contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """Wrap code in modern PyTorch mixed precision (FP16) on CUDA."""
        if self.is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                yield
        else:
            yield

    def empty_cache(self) -> None:
        """Release unused cached memory held by the PyTorch CUDA caching allocator."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# Export a shared singleton instance
default_device_manager = DeviceManager()