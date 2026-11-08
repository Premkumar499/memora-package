"""Tests for MemoryMonitor and MemoryTracker."""

import time
import torch

from memora.hardware.memory_monitor import MemoryMonitor, MemoryTracker


def test_memory_monitor_lifecycle():
    monitor = MemoryMonitor(device=torch.device("cpu"))
    monitor.start()
    assert len(monitor.snapshots) >= 1

    # Take step samples
    s1 = monitor.sample(step=1)
    assert s1.step == 1
    assert s1.cpu_ram_used > 0

    s2 = monitor.sample(step=2)
    assert s2.step == 2
    assert len(monitor.snapshots) == 3

    summary = monitor.stop()
    assert summary["snapshots_count"] == 3
    assert summary["peak_cpu_ram"] > 0


def test_memory_tracker_context():
    with MemoryTracker(device=torch.device("cpu")) as tracker:
        # Allocate some memory
        data = torch.randn(1000, 1000)
        time.sleep(0.01)

    assert tracker.duration_seconds > 0.0
    assert tracker.peak_allocated > 0

