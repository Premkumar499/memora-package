"""Tests for hardware discovery, GPU detection, and CPU fallbacks."""

from unittest.mock import MagicMock, patch
import pytest
import torch

from memora.hardware.gpu_detector import DeviceInfo, GPUDetector, HardwareReport
from memora.utils.common import format_bytes, resolve_device


def test_format_bytes():
    assert format_bytes(500) == "500.00 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
    assert format_bytes(0) == "0.00 B"
    assert format_bytes(None) == "N/A"
    assert format_bytes(-1024) == "-1.00 KB"


def test_resolve_device():
    # CPU resolution
    dev_cpu = resolve_device("cpu")
    assert dev_cpu.type == "cpu"

    # Auto resolution
    dev_auto = resolve_device("auto")
    assert dev_auto.type in ("cpu", "cuda", "mps")

    # Invalid device string
    with pytest.raises(Exception):
        resolve_device("non_existent_accelerator_device_xyz")


def test_cpu_detection():
    dev_info = GPUDetector.get_device_info(torch.device("cpu"))
    assert dev_info.device_type == "cpu"
    assert dev_info.device_index == 0
    assert dev_info.total_memory > 0
    assert dev_info.is_available is True
    assert "cores" in dev_info.name


def test_hardware_report():
    report = GPUDetector.get_hardware_report()
    assert isinstance(report, HardwareReport)
    assert report.python_version != ""
    assert report.torch_version != ""
    assert report.device_count >= 1
    assert report.system_ram_total > 0

    report_dict = report.to_dict()
    assert "platform_system" in report_dict
    assert "default_device" in report_dict


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 3050 6GB Laptop GPU")
@patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 6 * 1024**3))
def test_mocked_cuda_detection(mock_mem, mock_name, mock_avail):
    mock_props = MagicMock()
    mock_props.total_memory = 6 * 1024**3
    mock_props.major = 8
    mock_props.minor = 6

    with patch("torch.cuda.get_device_properties", return_value=mock_props):
        dev = GPUDetector.get_device_info(torch.device("cuda:0"))
        assert dev.device_type == "cuda"
        assert dev.name == "NVIDIA RTX 3050 6GB Laptop GPU"
        assert dev.total_memory == 6 * 1024**3
        assert dev.compute_capability == "8.6"
        assert dev.supports_bf16 is True
        assert dev.supports_fp16 is True

