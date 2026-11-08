"""Structured run report generator and comparative analysis."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table

from memora.utils.common import format_bytes


@dataclass
class RunReport:
    """Complete summary report of an optimization and training run."""

    run_id: str
    timestamp: str
    hardware_info: Dict[str, Any]
    config: Dict[str, Any]
    optimization_decisions: Dict[str, Any]
    training_metrics: Dict[str, Any]
    profiler_summary: Dict[str, Any]
    step_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "hardware_info": self.hardware_info,
            "config": self.config,
            "optimization_decisions": self.optimization_decisions,
            "training_metrics": self.training_metrics,
            "profiler_summary": self.profiler_summary,
            "step_metrics": self.step_metrics,
        }

    def save_json(self, filepath: str) -> str:
        """Save report to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

    def to_markdown(self) -> str:
        """Format report as Markdown."""
        lines = [
            f"# Memora Training & Optimization Report: `{self.run_id}`",
            f"- **Timestamp**: {self.timestamp}",
            f"- **Device**: {self.hardware_info.get('default_device', {}).get('name', 'Unknown')}",
            f"- **PyTorch Version**: {self.hardware_info.get('torch_version', 'Unknown')}",
            "",
            "## Optimization Decisions",
            f"- **Batch Size**: {self.optimization_decisions.get('batch_size')}",
            f"- **Precision**: {self.optimization_decisions.get('precision')} (GradScaler: {self.optimization_decisions.get('uses_grad_scaler')})",
            f"- **Gradient Accumulation**: {self.optimization_decisions.get('gradient_accumulation_steps')} steps (Effective Batch Size: {self.optimization_decisions.get('effective_batch_size')})",
            f"- **Activation Checkpointing**: {self.optimization_decisions.get('activation_checkpointing')}",
            "",
            "### Justifications",
        ]

        reasons = self.optimization_decisions.get("reasons", {})
        for opt_key, reason_str in reasons.items():
            lines.append(f"- **{opt_key.capitalize()}**: {reason_str}")

        lines.extend([
            "",
            "## Performance & Memory Metrics",
            f"- **Peak Memory**: {self.profiler_summary.get('overall_peak_memory_str', 'N/A')}",
            f"- **Mean Step Time**: {self.profiler_summary.get('mean_step_time_seconds', 'N/A')} s",
            f"- **Throughput**: {self.profiler_summary.get('mean_throughput_samples_per_sec', 'N/A')} samples/sec",
            f"- **Final Training Loss**: {self.training_metrics.get('final_loss', 'N/A')}",
            f"- **Final Accuracy**: {self.training_metrics.get('final_accuracy', 'N/A')}%",
        ])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print rich summary table to console."""
        console = Console()
        table = Table(title=f"Memora Run Summary [{self.run_id}]", header_style="bold cyan")
        table.add_column("Optimization / Metric", style="bold")
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")

        table.add_row(
            "Device",
            str(self.hardware_info.get("default_device", {}).get("name", "Unknown")),
            f"CUDA: {self.hardware_info.get('cuda_available')}",
        )
        table.add_row(
            "Batch Size",
            str(self.optimization_decisions.get("batch_size")),
            f"Accumulation: {self.optimization_decisions.get('gradient_accumulation_steps')} steps",
        )
        table.add_row(
            "Precision",
            str(self.optimization_decisions.get("precision")),
            f"Scaler: {self.optimization_decisions.get('uses_grad_scaler')}",
        )
        table.add_row(
            "Activation Checkpointing",
            str(self.optimization_decisions.get("activation_checkpointing")),
            self.optimization_decisions.get("reasons", {}).get("checkpointing", ""),
        )
        table.add_row(
            "Peak Memory",
            str(self.profiler_summary.get("overall_peak_memory_str", "N/A")),
            f"Bytes: {self.profiler_summary.get('overall_peak_memory_bytes', 'N/A')}",
        )
        table.add_row(
            "Mean Throughput",
            f"{self.profiler_summary.get('mean_throughput_samples_per_sec', 'N/A')} smp/s",
            f"Mean Step Time: {self.profiler_summary.get('mean_step_time_seconds', 'N/A')}s",
        )
        console.print(table)


def compare_runs(baseline: RunReport, optimized: RunReport) -> Dict[str, Any]:
    """Compare a baseline run with an optimized Memora run."""
    b_mem = baseline.profiler_summary.get("overall_peak_memory_bytes") or 0
    o_mem = optimized.profiler_summary.get("overall_peak_memory_bytes") or 0

    b_thru = baseline.profiler_summary.get("mean_throughput_samples_per_sec") or 0.0
    o_thru = optimized.profiler_summary.get("mean_throughput_samples_per_sec") or 0.0

    mem_saved_pct = round(((b_mem - o_mem) / b_mem * 100), 2) if b_mem > 0 else 0.0
    thru_speedup_pct = round(((o_thru - b_thru) / b_thru * 100), 2) if b_thru > 0 else 0.0

    return {
        "baseline_peak_memory": format_bytes(b_mem),
        "optimized_peak_memory": format_bytes(o_mem),
        "memory_saved_percent": mem_saved_pct,
        "baseline_throughput": b_thru,
        "optimized_throughput": o_thru,
        "throughput_change_percent": thru_speedup_pct,
    }

