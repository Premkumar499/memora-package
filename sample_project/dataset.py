"""Synthetic dataset generator for financial transactions risk detection."""

import torch
from torch.utils.data import TensorDataset


def get_risk_dataset(num_samples: int = 1000, num_features: int = 32):
    """Generate synthetic risk classification dataset."""
    torch.manual_seed(42)
    # Generate feature matrix
    features = torch.randn(num_samples, num_features)
    # Generate binary labels (0 = Normal, 1 = High Risk)
    labels = torch.randint(0, 2, (num_samples,))

    train_data = TensorDataset(features[:800], labels[:800])
    val_data = TensorDataset(features[800:], labels[800:])
    return train_data, val_data
