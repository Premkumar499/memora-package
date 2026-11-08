"""Deployment smoke test & health check script for Memora."""

import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from memora import Memora, GPUDetector, __version__


def run_health_check() -> int:
    print(f"=== Memora Deployment Health Check [v{__version__}] ===")

    # 1. Hardware & Runtime Diagnostics
    print("[1/3] Scanning Hardware & Environment...")
    hw = GPUDetector.get_hardware_report()
    print(f"  - Operating System  : {hw.platform_system} ({hw.platform_release})")
    print(f"  - Python Version    : {hw.python_version}")
    print(f"  - PyTorch Version   : {hw.torch_version}")
    print(f"  - CUDA Available    : {hw.cuda_available}")
    print(f"  - Compute Device    : {hw.default_device.name}")
    print(f"  - Total Device RAM  : {hw.to_dict()['default_device']['total_memory']} bytes")

    # 2. Synthetic Workload Probing
    print("[2/3] Testing Optimization Prober & Engine...")
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    dataset = TensorDataset(torch.randn(32, 16), torch.randint(0, 4, (32,)))

    trainer = Memora(
        model=model,
        dataset=dataset,
        batch_size="auto",
        precision="auto",
        gradient_accumulation="auto",
        activation_checkpointing="auto",
        device="auto",
        report_output_dir=None,
    )

    recs = trainer.optimize()
    print(f"  - Batch Size Recommendation  : {recs.recommended_batch_size}")
    print(f"  - Precision Recommendation   : {recs.recommended_precision}")
    print(f"  - Accumulation Steps         : {recs.recommended_grad_accum_steps}")

    # 3. Execution Verification
    print("[3/3] Executing Verification Training Step...")
    report = trainer.fit(epochs=1)
    if report.training_metrics.get("total_steps", 0) > 0:
        print(f"  - Steps Completed            : {report.training_metrics['total_steps']}")
        print(f"  - Mean Throughput            : {report.profiler_summary.get('mean_throughput_samples_per_sec')} smp/s")
        print("\n>>> ALL HEALTH CHECKS PASSED. Memora is ready for production deployment.")
        return 0
    else:
        print("\n>>> ERROR: Training step did not complete.")
        return 1


if __name__ == "__main__":
    sys.exit(run_health_check())

