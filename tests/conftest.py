"""Pytest fixtures for Memora tests."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for unit testing."""

    def __init__(self, in_features: int = 16, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class SimpleConvNet(nn.Module):
    """Simple 2D CNN with BatchNorm for testing vision workflows."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


@pytest.fixture
def mlp_model() -> nn.Module:
    return SimpleMLP()


@pytest.fixture
def conv_model() -> nn.Module:
    return SimpleConvNet()


@pytest.fixture
def synthetic_dataset() -> TensorDataset:
    num_samples = 64
    x = torch.randn(num_samples, 16)
    y = torch.randint(0, 4, (num_samples,))
    return TensorDataset(x, y)


@pytest.fixture
def image_dataset() -> TensorDataset:
    num_samples = 32
    x = torch.randn(num_samples, 3, 16, 16)
    y = torch.randint(0, 4, (num_samples,))
    return TensorDataset(x, y)

