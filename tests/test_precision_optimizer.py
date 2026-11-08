"""Tests for precision selection and autocast management."""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from memora.optimizer.precision_optimizer import PrecisionOptimizer


def test_cpu_precision_defaults_to_fp32():
    opt = PrecisionOptimizer(device=torch.device("cpu"), mode="auto")
    assert opt.selected_precision == "fp32"
    assert opt.use_scaler is False
    assert opt.scaler is None

    # Autocast context should execute without error
    with opt.autocast_context():
        x = torch.randn(2, 4)
        y = x * 2.0
    assert y.dtype == torch.float32


def test_explicit_fp32_mode():
    opt = PrecisionOptimizer(device=torch.device("cpu"), mode="fp32")
    assert opt.selected_precision == "fp32"
    assert opt.use_scaler is False


@patch("memora.hardware.gpu_detector.GPUDetector.get_device_info")
def test_cuda_ampere_selects_bf16(mock_info):
    mock_dev = MagicMock()
    mock_dev.supports_bf16 = True
    mock_dev.supports_fp16 = True
    mock_info.return_value = mock_dev

    opt = PrecisionOptimizer(device=torch.device("cuda:0"), mode="auto")
    assert opt.selected_precision == "bf16"
    assert opt.use_scaler is False


@patch("memora.hardware.gpu_detector.GPUDetector.get_device_info")
def test_cuda_turing_selects_fp16_with_scaler(mock_info):
    mock_dev = MagicMock()
    mock_dev.supports_bf16 = False
    mock_dev.supports_fp16 = True
    mock_info.return_value = mock_dev

    opt = PrecisionOptimizer(device=torch.device("cuda:0"), mode="auto")
    assert opt.selected_precision == "fp16"
    assert opt.use_scaler is True


def test_backward_and_step_fp32(mlp_model):
    opt = PrecisionOptimizer(device=torch.device("cpu"), mode="fp32")
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=0.01)

    x = torch.randn(4, 16)
    out = mlp_model(x)
    loss = out.sum()

    opt.backward(loss)
    opt.step(optimizer)
    assert any(p.grad is not None for p in mlp_model.parameters())
