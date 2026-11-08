"""Training profiler for capturing forward, backward, step time, memory, and throughput."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import psutil
import torch

from memora.utils.common import format_bytes


@dataclass
class StepMetric:
    """Metrics recorded for a single training step."""

    step: int
    batch_size: int
    forward_time: float
    backward_time: float
    step_time: float
    throughput_samples_per_sec: float
    peak_memory_bytes: int
    cpu_ram_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "batch_size": self.batch_size,
            "forward_time": round(self.forward_time, 5),
            "backward_time": round(self.backward_time, 5),
            "step_time": round(self.step_time, 5),
            "throughput": round(self.throughput_samples_per_sec, 2),
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_str": format_bytes(self.peak_memory_bytes),
            "cpu_ram_bytes": self.cpu_ram_bytes,
            "cpu_ram_str": format_bytes(self.cpu_ram_bytes),
        }


class TrainingProfiler:
    """Measures step-level execution time, memory usage, and throughput during training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.step_metrics: List[StepMetric] = []
        self._current_step = 0
        self._t_start: float = 0.0
        self._t_forward: float = 0.0
        self._t_backward: float = 0.0

    def start_step(self, step: int) -> None:
        """Mark start of a training step."""
        self._current_step = step
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t_start = time.perf_counter()

    def mark_forward_done(self) -> None:
        """Mark completion of forward pass."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._t_forward = time.perf_counter()

    def end_step(self, batch_size: int) -> StepMetric:
        """Mark end of backward pass and optimizer step, recording metrics."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        t_end = time.perf_counter()

        step_time = max(1e-6, t_end - self._t_start)
        forward_time = max(0.0, self._t_forward - self._t_start) if self._t_forward > self._t_start else 0.0
        backward_time = max(0.0, t_end - self._t_forward) if self._t_forward > 0 else step_time

        throughput = batch_size / step_time

        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_mem = psutil.Process().memory_info().rss

        cpu_ram = psutil.virtual_memory().used

        metric = StepMetric(
            step=self._current_step,
            batch_size=batch_size,
            forward_time=forward_time,
            backward_time=backward_time,
            step_time=step_time,
            throughput_samples_per_sec=throughput,
            peak_memory_bytes=peak_mem,
            cpu_ram_bytes=cpu_ram,
        )
        self.step_metrics.append(metric)
        return metric

    def summary(self) -> Dict[str, Any]:
        """Aggregate summary over all recorded steps."""
        if not self.step_metrics:
            return {}

        # Discard first step if warmup
        records = self.step_metrics[1:] if len(self.step_metrics) > 1 else self.step_metrics
        step_times = [m.step_time for m in records]
        throughputs = [m.throughput_samples_per_sec for m in records]
        peak_mems = [m.peak_memory_bytes for m in self.step_metrics]

        mean_step_time = sum(step_times) / len(step_times)
        mean_throughput = sum(throughputs) / len(throughputs)
        overall_peak_mem = max(peak_mems)

        return {
            "total_steps": len(self.step_metrics),
            "measured_steps": len(records),
            "mean_step_time_seconds": round(mean_step_time, 5),
            "mean_throughput_samples_per_sec": round(mean_throughput, 2),
            "overall_peak_memory_bytes": overall_peak_mem,
            "overall_peak_memory_str": format_bytes(overall_peak_mem),
        }

