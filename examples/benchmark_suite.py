"""Reproducible benchmark suite for evaluating Memora across workloads."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from memora import Memora, GPUDetector, compare_runs
from memora.reporting.report import RunReport


class MLPBenchmark(nn.Module):
    def __init__(self, in_features: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBenchmark(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def run_workload_benchmark(
    name: str,
    model_fn,
    dataset: TensorDataset,
    baseline_batch_size: int,
    epochs: int = 2,
) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f" Running Benchmark: {name}")
    print(f"==========================================")

    # 1. Baseline Run
    print(f"[*] Running Baseline (fixed batch_size={baseline_batch_size})...")
    baseline_model = model_fn()
    baseline_trainer = Memora(
        model=baseline_model,
        dataset=dataset,
        batch_size=baseline_batch_size,
        precision="fp32",
        gradient_accumulation=1,
        activation_checkpointing=False,
    )
    baseline_report = baseline_trainer.fit(epochs=epochs)

    # 2. Memora Optimized Run
    print(f"[*] Running Memora Auto-Optimization...")
    opt_model = model_fn()
    memora_trainer = Memora(
        model=opt_model,
        dataset=dataset,
        batch_size="auto",
        precision="auto",
        gradient_accumulation="auto",
        activation_checkpointing="auto",
        effective_batch_size=baseline_batch_size * 2,
    )
    opt_report = memora_trainer.fit(epochs=epochs)

    # 3. Compare
    comparison = compare_runs(baseline_report, opt_report)
    print(f"[+] Completed: Baseline throughput {comparison['baseline_throughput']} smp/s -> Optimized {comparison['optimized_throughput']} smp/s ({comparison['throughput_change_percent']}%)")

    return {
        "workload": name,
        "baseline_batch_size": baseline_batch_size,
        "memora_batch_size": opt_report.optimization_decisions.get("batch_size"),
        "memora_precision": opt_report.optimization_decisions.get("precision"),
        "memora_accum_steps": opt_report.optimization_decisions.get("gradient_accumulation_steps"),
        "baseline_peak_memory": comparison["baseline_peak_memory"],
        "optimized_peak_memory": comparison["optimized_peak_memory"],
        "memory_saved_percent": comparison["memory_saved_percent"],
        "baseline_throughput": comparison["baseline_throughput"],
        "optimized_throughput": comparison["optimized_throughput"],
        "throughput_change_percent": comparison["throughput_change_percent"],
    }


def main():
    os.makedirs("./benchmarks", exist_ok=True)
    hw_report = GPUDetector.get_hardware_report()

    print("=== Memora Comprehensive Benchmark Suite ===")
    print(f"Device: {hw_report.default_device.name}")
    print(f"Python: {hw_report.python_version} | PyTorch: {hw_report.torch_version}")

    results: List[Dict[str, Any]] = []

    # Workload 1: MLP Classification
    num_samples_mlp = 256
    mlp_dataset = TensorDataset(torch.randn(num_samples_mlp, 128), torch.randint(0, 10, (num_samples_mlp,)))
    r1 = run_workload_benchmark(
        name="MLP (Tabular/Dense)",
        model_fn=MLPBenchmark,
        dataset=mlp_dataset,
        baseline_batch_size=16,
        epochs=2,
    )
    results.append(r1)

    # Workload 2: CNN Image Classification
    num_samples_cnn = 128
    cnn_dataset = TensorDataset(torch.randn(num_samples_cnn, 3, 32, 32), torch.randint(0, 10, (num_samples_cnn,)))
    r2 = run_workload_benchmark(
        name="CNN (Vision/Convolutional)",
        model_fn=CNNBenchmark,
        dataset=cnn_dataset,
        baseline_batch_size=8,
        epochs=2,
    )
    results.append(r2)

    # Save summary results
    summary_path = "./benchmarks/benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"hardware": hw_report.to_dict(), "results": results}, f, indent=2)
    print(f"\nSaved benchmark results to {summary_path}")

    # Format Markdown
    md_lines = [
        "# Memora Benchmark Results",
        f"- **Hardware**: {hw_report.default_device.name}",
        f"- **CUDA Available**: {hw_report.cuda_available}",
        f"- **PyTorch**: {hw_report.torch_version}",
        "",
        "| Workload | Baseline Batch | Memora Batch | Baseline Throughput | Optimized Throughput | Throughput Gain |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        md_lines.append(
            f"| {r['workload']} | {r['baseline_batch_size']} | {r['memora_batch_size']} | "
            f"{r['baseline_throughput']} smp/s | {r['optimized_throughput']} smp/s | {r['throughput_change_percent']}% |"
        )

    md_report_path = "./benchmarks/BENCHMARKS.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Generated Markdown report at {md_report_path}")


if __name__ == "__main__":
    main()
