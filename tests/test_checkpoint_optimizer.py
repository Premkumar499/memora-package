"""Tests for activation checkpointing optimizer."""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from memora.optimizer.checkpoint_optimizer import CheckpointOptimizer, CheckpointedSequential


def test_checkpointed_sequential_forward_backward():
    layer1 = nn.Linear(8, 16)
    layer2 = nn.Linear(16, 4)
    model = CheckpointedSequential(layer1, layer2)
    model.train()

    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 4)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert layer1.weight.grad is not None
    assert layer2.weight.grad is not None


def test_checkpoint_optimizer_explicit_flags(mlp_model):
    opt_true = CheckpointOptimizer(model=mlp_model, enabled=True)
    assert opt_true.is_enabled is True

    opt_false = CheckpointOptimizer(model=mlp_model, enabled=False)
    assert opt_false.is_enabled is False


@patch("torch.cuda.is_available", return_value=True)
def test_checkpoint_auto_decision_low_vram(mock_avail, mlp_model):
    mock_props = MagicMock()
    mock_props.total_memory = 6 * 1024**3  # 6GB <= 8.5GB

    with patch("torch.cuda.get_device_properties", return_value=mock_props):
        opt = CheckpointOptimizer(
            model=mlp_model,
            enabled="auto",
            device=torch.device("cuda:0"),
        )
        assert opt.is_enabled is True
        assert "6.0GB" in opt.reason

