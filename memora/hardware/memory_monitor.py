"""Memory monitor for tracking GPU and CPU memory during training and probing."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import psutil
import torch

from memora.hardware.gpu_detector import GPUDetector
from memora.utils.common import format_bytes


@dataclass
class MemorySnapshot:
    """A point-in-time snapshot of system and device memory."""

    timestamp: float
    step: Optional[int] = None
    cuda_allocated: int = 0  # bytes
    cuda_reserved: int = 0  # bytes
    cuda_max_allocated: int = 0  # bytes
    cuda_max_reserved: int = 0  # bytes
    cpu_ram_used: int = 0  # bytes
    cpu_ram_percent: float = 0.0
    device_free_memory: Optional[int] = None  # bytes
    nvml_device_used: Optional[int] = None  # bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary with readable strings."""
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "cuda_allocated": self.cuda_allocated,
            "cuda_allocated_str": format_bytes(self.cuda_allocated),
            "cuda_reserved": self.cuda_reserved,
            "cuda_reserved_str": format_bytes(self.cuda_reserved),
            "cuda_max_allocated": self.cuda_max_allocated,
            "cuda_max_allocated_str": format_bytes(self.cuda_max_allocated),
            "cuda_max_reserved": self.cuda_max_reserved,
            "cuda_max_reserved_str": format_bytes(self.cuda_max_reserved),
            "cpu_ram_used": self.cpu_ram_used,
            "cpu_ram_used_str": format_bytes(self.cpu_ram_used),
            "cpu_ram_percent": self.cpu_ram_percent,
            "device_free_memory": self.device_free_memory,
            "nvml_device_used": self.nvml_device_used,
        }


class MemoryMonitor:
    """Continuously monitors memory snapshots during model training and execution."""

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.snapshots: List[MemorySnapshot] = []
        self._is_monitoring = False
        self._start_time: Optional[float] = None

    def reset_stats(self) -> None:
        """Clear memory cache and reset peak tracking."""
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.snapshots.clear()

    def start(self) -> None:
        """Start monitoring session."""
        self.reset_stats()
        self._is_monitoring = True
        self._start_time = time.time()
        self.sample(step=0)

    def sample(self, step: Optional[int] = None) -> MemorySnapshot:
        """Capture a point-in-time snapshot of memory."""
        now = time.time()
        vm = psutil.virtual_memory()

        cuda_alloc = 0
        cuda_res = 0
        cuda_max_alloc = 0
        cuda_max_res = 0
        free_mem = None
        nvml_used = None

        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_alloc = torch.cuda.memory_allocated(self.device)
            cuda_res = torch.cuda.memory_reserved(self.device)
            cuda_max_alloc = torch.cuda.max_memory_allocated(self.device)
            cuda_max_res = torch.cuda.max_memory_reserved(self.device)
            try:
                free_m, _ = torch.cuda.mem_get_info(self.device)
                free_mem = free_m
            except Exception:
                pass

            dev_idx = self.device.index or 0
            nvml_info = GPUDetector.get_nvml_memory(dev_idx)
            if nvml_info:
                nvml_used = nvml_info.get("used")

        snapshot = MemorySnapshot(
            timestamp=now,
            step=step,
            cuda_allocated=cuda_alloc,
            cuda_reserved=cuda_res,
            cuda_max_allocated=cuda_max_alloc,
            cuda_max_reserved=cuda_max_res,
            cpu_ram_used=vm.used,
            cpu_ram_percent=vm.percent,
            device_free_memory=free_mem,
            nvml_device_used=nvml_used,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and generate summary statistics."""
        self._is_monitoring = False
        if not self.snapshots:
            return {}

        peak_allocated = max(s.cuda_allocated for s in self.snapshots)
        peak_reserved = max(s.cuda_reserved for s in self.snapshots)
        peak_cpu_ram = max(s.cpu_ram_used for s in self.snapshots)
        peak_max_allocated = max(s.cuda_max_allocated for s in self.snapshots)

        return {
            "snapshots_count": len(self.snapshots),
            "peak_cuda_allocated": peak_allocated,
            "peak_cuda_allocated_str": format_bytes(peak_allocated),
            "peak_cuda_reserved": peak_reserved,
            "peak_cuda_reserved_str": format_bytes(peak_reserved),
            "peak_cuda_max_allocated": peak_max_allocated,
            "peak_cuda_max_allocated_str": format_bytes(peak_max_allocated),
            "peak_cpu_ram": peak_cpu_ram,
            "peak_cpu_ram_str": format_bytes(peak_cpu_ram),
        }


class MemoryTracker:
    """Context manager for measuring memory consumed by a specific block of code."""

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.start_allocated: int = 0
        self.peak_allocated: int = 0
        self.end_allocated: int = 0
        self.allocated_delta: int = 0
        self.duration_seconds: float = 0.0

    def __enter__(self) -> MemoryTracker:
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_allocated = torch.cuda.memory_allocated(self.device)
        else:
            self.start_allocated = psutil.Process().memory_info().rss
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_seconds = time.perf_counter() - self._start_time
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.end_allocated = torch.cuda.memory_allocated(self.device)
            self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
            self.allocated_delta = max(0, self.peak_allocated - self.start_allocated)
        else:
            self.end_allocated = psutil.Process().memory_info().rss
            self.peak_allocated = self.end_allocated
            self.allocated_delta = max(0, self.end_allocated - self.start_allocated)

