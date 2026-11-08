"""Gradient accumulation manager for maintaining target effective batch size with constrained micro-batches."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union
import torch


class GradientAccumulationOptimizer:
    """Manages gradient accumulation steps, loss scaling, and optimizer step boundaries."""

    def __init__(
        self,
        micro_batch_size: int,
        effective_batch_size: Optional[int] = None,
        accumulation_steps: Union[int, str] = "auto",
    ):
        self.micro_batch_size = max(1, micro_batch_size)
        self.effective_batch_size = effective_batch_size
        self.accumulation_steps = self._compute_steps(accumulation_steps)
        self.current_accum_step = 0

    def _compute_steps(self, steps_param: Union[int, str]) -> int:
        """Compute the number of accumulation steps needed."""
        if isinstance(steps_param, int) and steps_param >= 1:
            return steps_param

        if self.effective_batch_size is not None and self.effective_batch_size > 0:
            steps = math.ceil(self.effective_batch_size / self.micro_batch_size)
            return max(1, steps)

        return 1

    @property
    def computed_effective_batch_size(self) -> int:
        """The actual effective batch size based on micro-batch and accumulation steps."""
        return self.micro_batch_size * self.accumulation_steps

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize loss by accumulation steps so accumulated gradients match the effective batch size."""
        if self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss

    def should_step(self, batch_idx: int, total_batches: Optional[int] = None) -> bool:
        """Determine whether optimizer.step() should occur at this batch index.

        Triggers on:
        1. Every N-th step where (batch_idx + 1) % accumulation_steps == 0.
        2. At the final batch of the epoch to flush remaining gradients.
        """
        is_step_boundary = (batch_idx + 1) % self.accumulation_steps == 0
        is_final_batch = total_batches is not None and (batch_idx + 1) == total_batches
        return is_step_boundary or is_final_batch

    def reset(self) -> None:
        """Reset internal accumulation counters."""
        self.current_accum_step = 0

    def to_dict(self) -> Dict[str, Any]:
        """Summary of accumulation parameters."""
        return {
            "micro_batch_size": self.micro_batch_size,
            "target_effective_batch_size": self.effective_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "computed_effective_batch_size": self.computed_effective_batch_size,
        }
