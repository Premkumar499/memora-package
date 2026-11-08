"""Image classification demonstration comparing baseline PyTorch vs Memora optimizations."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from memora import Memora, compare_runs


class ConvNet(nn.Module):
    """Fallback convolutional neural network when torchvision is not used."""

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
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_model(num_classes: int = 10) -> nn.Module:
    if HAS_TORCHVISION:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return ConvNet(num_classes=num_classes)


def main():
    print("=== Memora Image Classification Benchmark & Comparison ===")

    # Generate synthetic image dataset (3 channels, 32x32 pixels, 10 classes)
    num_samples = 128
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)

    # 1. Baseline Run (Manual fixed batch size, no auto-tuning)
    print("\n[Phase 1] Executing Baseline PyTorch Training Run...")
    baseline_model = get_model()
    baseline_trainer = Memora(
        model=baseline_model,
        dataset=dataset,
        batch_size=16,
        precision="fp32",
        gradient_accumulation=1,
        activation_checkpointing=False,
    )
    baseline_report = baseline_trainer.fit(epochs=2)
    print("Baseline run completed.")

    # 2. Memora Optimized Run
    print("\n[Phase 2] Executing Memora Auto-Optimized Run...")
    optimized_model = get_model()
    memora_trainer = Memora(
        model=optimized_model,
        dataset=dataset,
        batch_size="auto",
        precision="auto",
        gradient_accumulation="auto",
        activation_checkpointing="auto",
        effective_batch_size=32,
    )
    optimized_report = memora_trainer.fit(epochs=2)
    print("Optimized run completed.")

    # 3. Comparative Analysis
    print("\n=== Comparative Analysis: Baseline vs Memora ===")
    comparison = compare_runs(baseline_report, optimized_report)
    for metric, val in comparison.items():
        print(f"  {metric:<30}: {val}")

    print("\n--- Optimized Run Summary Table ---")
    optimized_report.print_summary()


if __name__ == "__main__":
    main()

