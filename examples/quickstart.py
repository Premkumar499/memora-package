"""Quickstart example showing Memora automatic memory optimization."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from memora import Memora


class ToyClassifier(nn.Module):
    """A simple multi-layer perceptron for demonstration."""

    def __init__(self, in_features: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    print("=== Memora Quickstart Demonstration ===")

    # 1. Prepare synthetic dataset
    num_samples = 256
    features = torch.randn(num_samples, 64)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(features, labels)

    model = ToyClassifier()

    # 2. Instantiate Memora with automatic memory optimization
    trainer = Memora(
        model=model,
        dataset=dataset,
        batch_size="auto",
        precision="auto",
        gradient_accumulation="auto",
        activation_checkpointing="auto",
        device="auto",
    )

    # 3. Analyze workload and preview recommendations
    print("\n--- Running Optimization Analysis ---")
    recommendations = trainer.optimize()
    print(f"Recommended Batch Size       : {recommendations.recommended_batch_size}")
    print(f"Recommended Precision        : {recommendations.recommended_precision}")
    print(f"Gradient Accumulation Steps  : {recommendations.recommended_grad_accum_steps}")
    print(f"Activation Checkpointing     : {recommendations.recommended_checkpointing}")

    # 4. Train model with selected optimizations
    print("\n--- Training Model ---")
    report = trainer.fit(epochs=2)

    # 5. Print summary
    print("\n--- Execution Summary ---")
    report.print_summary()


if __name__ == "__main__":
    main()
