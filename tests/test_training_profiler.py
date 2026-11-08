"""Tests for TrainingProfiler and step metrics."""

import time
import torch

from memora.profiler.training_profiler import TrainingProfiler


def test_profiler_steps_and_summary():
    device = torch.device("cpu")
    profiler = TrainingProfiler(device=device)

    for step in range(1, 4):
        profiler.start_step(step)
        time.sleep(0.005)
        profiler.mark_forward_done()
        time.sleep(0.005)
        metric = profiler.end_step(batch_size=8)
        assert metric.step == step
        assert metric.batch_size == 8
        assert metric.step_time > 0.0
        assert metric.throughput_samples_per_sec > 0.0

    assert len(profiler.step_metrics) == 3
    summary = profiler.summary()
    assert summary["total_steps"] == 3
    assert summary["mean_step_time_seconds"] > 0
    assert summary["mean_throughput_samples_per_sec"] > 0
