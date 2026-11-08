"""Mixed precision optimizer and autocast manager."""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn

from memora.hardware.gpu_detector import GPUDetector


PrecisionMode = Literal["auto", "fp32", "fp16", "bf16"]


class PrecisionOptimizer:
    """Selects and manages mixed-precision training (FP32, FP16, BF16) with AMP GradScaler."""

    def __init__(
        self,
        device: torch.device,
        mode: PrecisionMode = "auto",
        enabled: Optional[bool] = None,
    ):
        self.device = device
        self.requested_mode = mode
        self.enabled = enabled

        self.selected_precision, self.use_scaler, self.reason = self._determine_precision()
        self.scaler = self._create_scaler()

    def _determine_precision(self) -> Tuple[str, bool, str]:
        """Determine precision mode, whether GradScaler is needed, and reasoning."""
        dev_info = GPUDetector.get_device_info(self.device)

        if self.enabled is False or self.requested_mode == "fp32":
            return "fp32", False, "User selected FP32 precision (AMP disabled)."

        # CUDA device
        if self.device.type == "cuda":
            if self.requested_mode == "bf16":
                if dev_info.supports_bf16:
                    return "bf16", False, "Hardware supports BF16 native operations without GradScaler."
                return "fp16", True, "BF16 requested but unsupported on this CUDA compute capability; fell back to FP16 with GradScaler."

            if self.requested_mode == "fp16":
                return "fp16", True, "User requested FP16 precision with GradScaler."

            # Auto mode on CUDA
            if dev_info.supports_bf16:
                return "bf16", False, "Hardware supports native BF16 (Compute Capability >= 8.0); selected BF16 for speed and stability without scaling."
            elif dev_info.supports_fp16:
                return "fp16", True, "Hardware supports FP16; selected FP16 with AMP GradScaler to reduce VRAM consumption."
            return "fp32", False, "CUDA hardware does not support fast half precision; selected FP32."

        # MPS (Apple Silicon)
        if self.device.type == "mps":
            if self.requested_mode in ("fp16", "auto"):
                return "fp16", False, "Apple Silicon MPS supports FP16 autocast."
            return "fp32", False, "FP32 selected for MPS."

        # CPU device
        if self.requested_mode == "bf16" and hasattr(torch, "bfloat16"):
            return "bf16", False, "CPU supports bfloat16 autocast."

        return "fp32", False, "Running on CPU; selected standard FP32 precision."

    def _create_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Instantiate GradScaler if required."""
        if self.use_scaler and self.device.type == "cuda" and torch.cuda.is_available():
            return torch.amp.GradScaler("cuda", enabled=True)
        return None

    @contextlib.contextmanager
    def autocast_context(self) -> Iterator[None]:
        """Context manager applying the configured autocast."""
        if self.selected_precision == "fp32":
            # Disabled autocast
            yield
            return

        device_type = self.device.type
        if device_type not in ("cuda", "cpu"):
            device_type = "cpu"

        dtype = torch.bfloat16 if self.selected_precision == "bf16" else torch.float16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
            yield

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with scaler if active, else standard loss.backward()."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with scaler if active, else standard optimizer.step()."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def to_dict(self) -> Dict[str, Any]:
        """Summary of precision configuration."""
        return {
            "requested_mode": self.requested_mode,
            "selected_precision": self.selected_precision,
            "uses_grad_scaler": self.use_scaler,
            "reason": self.reason,
        }

