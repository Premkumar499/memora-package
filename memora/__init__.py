"""Memora: A memory-aware training optimization framework for PyTorch."""

from memora.config import MemoraConfig, OptimizationRecommendation
from memora.engine import Memora
from memora.hardware.gpu_detector import DeviceInfo, GPUDetector, HardwareReport
from memora.hardware.memory_monitor import MemoryMonitor, MemorySnapshot, MemoryTracker
from memora.reporting.report import RunReport, compare_runs

__version__ = "0.1.0"

__all__ = [
    "Memora",
    "MemoraConfig",
    "OptimizationRecommendation",
    "GPUDetector",
    "DeviceInfo",
    "HardwareReport",
    "MemoryMonitor",
    "MemorySnapshot",
    "MemoryTracker",
    "RunReport",
    "compare_runs",
]
