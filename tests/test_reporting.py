"""Tests for RunReport and comparison utilities."""

import os
import tempfile
from memora.reporting.report import RunReport, compare_runs


def test_run_report_export_and_json():
    report = RunReport(
        run_id="test_run_123",
        timestamp="2026-11-08 20:00:00",
        hardware_info={"default_device": {"name": "Test Device"}, "cuda_available": False},
        config={"batch_size": 16},
        optimization_decisions={"batch_size": 16, "precision": "fp32", "uses_grad_scaler": False},
        training_metrics={"epochs": 1, "final_loss": 0.42},
        profiler_summary={"overall_peak_memory_bytes": 1024 * 1024 * 50, "mean_throughput_samples_per_sec": 100.0},
    )

    # Test Markdown formatting
    md = report.to_markdown()
    assert "# Memora Training & Optimization Report" in md
    assert "Test Device" in md

    # Test JSON file persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        saved_path = report.save_json(json_path)
        assert os.path.exists(saved_path)
        with open(saved_path, "r") as f:
            content = f.read()
        assert "test_run_123" in content


def test_compare_runs():
    base = RunReport(
        run_id="baseline",
        timestamp="2026-11-08",
        hardware_info={},
        config={},
        optimization_decisions={},
        training_metrics={},
        profiler_summary={"overall_peak_memory_bytes": 1000, "mean_throughput_samples_per_sec": 50.0},
    )
    opt = RunReport(
        run_id="optimized",
        timestamp="2026-11-08",
        hardware_info={},
        config={},
        optimization_decisions={},
        training_metrics={},
        profiler_summary={"overall_peak_memory_bytes": 600, "mean_throughput_samples_per_sec": 75.0},
    )

    diff = compare_runs(base, opt)
    assert diff["memory_saved_percent"] == 40.0
    assert diff["throughput_change_percent"] == 50.0

