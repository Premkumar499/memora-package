"""Sample Deep Neural Network for Risk Classification."""

import torch
import torch.nn as nn


class RiskPredictionModel(nn.Module):
    """Multi-layer classifier with residual connections and batch normalization."""

    def __init__(self, in_features: int = 32, num_classes: int = 2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.input_layer(x)
        h2 = self.block1(h1) + h1  # Residual skip connection
        h3 = self.block2(h2)
        return self.classifier(h3)
