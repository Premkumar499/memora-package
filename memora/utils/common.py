"""Common utilities for device handling, memory formatting, and reproducibility."""

from __future__ import annotations

import os
import random
from typing import Union
import torch


def format_bytes(num_bytes: Union[int, float], precision: int = 2) -> str:
    """Format bytes into human-readable string (B, KB, MB, GB, TB)."""
    if num_bytes < 0:
        return f"-{format_bytes(-num_bytes, precision)}"
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.{precision}f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.{precision}f} EB"


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Resolve device specification to a concrete torch.device.

    Supports:
        - 'auto': chooses 'cuda' if CUDA is available, else 'mps' if Apple Silicon MPS is available, else 'cpu'.
        - 'cuda', 'cuda:X': returns CUDA device if CUDA is available, otherwise raises ValueError or falls back.
        - 'mps': returns MPS device if available.
        - 'cpu': returns CPU device.
    """
    if isinstance(device, torch.device):
        return device

    device_str = str(device).strip().lower()
    if device_str in ("auto", "default"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device_str}' is unavailable because CUDA is not supported or no GPU is present. "
                "Use device='cpu' or device='auto' instead."
            )
        return torch.device(device_str)

    if device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "Requested device 'mps' is unavailable. Use device='cpu' or device='auto' instead."
            )
        return torch.device("mps")

    if device_str == "cpu":
        return torch.device("cpu")

    return torch.device(device_str)


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, random, and PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
