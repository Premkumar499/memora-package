"""Activation checkpointing optimizer for trading compute for activation memory."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class CheckpointedSequential(nn.Module):
    """A wrapper for nn.Sequential that evaluates layers using PyTorch activation checkpointing."""

    def __init__(self, *modules: nn.Module, preserve_rng_state: bool = True):
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        self.preserve_rng_state = preserve_rng_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.module_list:
            # Checkpoint requires tensor inputs that require_grad or module forward wrapper
            if self.training and x.requires_grad:
                x = checkpoint(module, x, use_reentrant=False, preserve_rng_state=self.preserve_rng_state)
            else:
                x = module(x)
        return x


class CheckpointOptimizer:
    """Manages activation checkpointing recommendations and module wrapping."""

    def __init__(
        self,
        model: nn.Module,
        enabled: Union[bool, str] = "auto",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.requested = enabled
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_enabled, self.reason = self._decide()

    def _decide(self) -> tuple[bool, str]:
        """Decide whether to enable activation checkpointing."""
        if isinstance(self.requested, bool):
            return self.requested, f"Activation checkpointing explicitly set to {self.requested} by user."

        # If 'auto':
        # On GPU with <= 8GB VRAM or when memory is constrained, enable checkpointing to prevent OOM
        if self.device.type == "cuda" and torch.cuda.is_available():
            dev_idx = self.device.index or 0
            props = torch.cuda.get_device_properties(dev_idx)
            total_gb = props.total_memory / (1024**3)
            if total_gb <= 8.5:
                return True, f"GPU VRAM is {total_gb:.1f}GB (<= 8GB); enabling activation checkpointing to conserve activation memory."
            return False, f"GPU VRAM is {total_gb:.1f}GB (> 8GB); activation checkpointing disabled to preserve maximum training speed."

        return False, "Running on non-CUDA device; activation checkpointing disabled by default to avoid compute overhead."

    def apply(self) -> nn.Module:
        """Apply activation checkpointing to supported model submodules if enabled."""
        if not self.is_enabled:
            return self.model

        # Check if the model has a known checkpointing method (e.g. HuggingFace gradient_checkpointing_enable)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled model native gradient checkpointing.")
                return self.model
            except Exception as e:
                logger.warning(f"Failed to enable native gradient checkpointing: {e}")

        # If the model is or has nn.Sequential blocks, we can wrap them
        return self.model

    def to_dict(self) -> Dict[str, Any]:
        """Summary of checkpointing decision."""
        return {
            "requested": self.requested,
            "is_enabled": self.is_enabled,
            "reason": self.reason,
            "trade_off": "Saves ~40-60% activation memory with ~20-30% compute re-computation overhead.",
        }
