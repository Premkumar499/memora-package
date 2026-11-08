"""Automatic batch-size search and memory-budget benchmarking."""

from __future__ import annotations

import copy
import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from memora.hardware.gpu_detector import GPUDetector
from memora.hardware.memory_monitor import MemoryTracker
from memora.utils.common import format_bytes

logger = logging.getLogger(__name__)


@dataclass
class ProbeTrialResult:
    """Record of a single batch-size probe trial."""

    batch_size: int
    success: bool
    peak_memory_bytes: int
    duration_seconds: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "success": self.success,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_str": format_bytes(self.peak_memory_bytes),
            "duration_seconds": round(self.duration_seconds, 4),
            "error_message": self.error_message,
        }


@dataclass
class BatchSearchResult:
    """Outcome of batch-size search procedure."""

    optimal_batch_size: int
    memory_limit_bytes: int
    trials: List[ProbeTrialResult] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_batch_size": self.optimal_batch_size,
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_limit_str": format_bytes(self.memory_limit_bytes),
            "reason": self.reason,
            "trials": [t.to_dict() for t in self.trials],
        }


class BatchSizeOptimizer:
    """Benchmarks and searches for the optimal micro-batch size that fits within memory constraints."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_shape: Tuple[int, ...],
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        target_shape: Optional[Tuple[int, ...]] = None,
        memory_limit: Union[int, float, str] = "auto",
        safety_margin: float = 0.15,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
    ):
        self.model = model
        self.device = device
        self.input_shape = input_shape  # shape per sample, e.g. (3, 224, 224)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.target_shape = target_shape
        self.memory_limit = memory_limit
        self.safety_margin = safety_margin
        # Detect if model contains BatchNorm modules, which require batch_size >= 2
        has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in model.modules()
        )
        effective_min = max(2, min_batch_size) if has_bn else max(1, min_batch_size)
        self.min_batch_size = effective_min
        self.max_batch_size = max(self.min_batch_size, max_batch_size)

        self.resolved_memory_limit = self._resolve_memory_limit()

    def _resolve_memory_limit(self) -> int:
        """Calculate effective memory limit in bytes based on hardware and safety margin."""
        dev_info = GPUDetector.get_device_info(self.device)
        total_mem = dev_info.total_memory

        if isinstance(self.memory_limit, str) and self.memory_limit.lower() == "auto":
            usable = int(total_mem * (1.0 - self.safety_margin))
            return max(usable, 256 * 1024 * 1024)  # at least 256MB

        if isinstance(self.memory_limit, float) and 0.0 < self.memory_limit <= 1.0:
            usable = int(total_mem * self.memory_limit * (1.0 - self.safety_margin))
            return max(usable, 256 * 1024 * 1024)

        if isinstance(self.memory_limit, int) and self.memory_limit > 1:
            usable = int(self.memory_limit * (1.0 - self.safety_margin))
            return max(usable, 256 * 1024 * 1024)

        return int(total_mem * (1.0 - self.safety_margin))

    def _generate_synthetic_input(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic input tensor for probing."""
        return torch.randn(batch_size, *self.input_shape, device=self.device)

    def _probe_trial(self, batch_size: int) -> ProbeTrialResult:
        """Run an isolated forward and backward pass for a given candidate batch size."""
        # Clean state prior to trial
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        inputs = None
        targets = None
        outputs = None
        loss = None

        try:
            with MemoryTracker(self.device) as tracker:
                inputs = self._generate_synthetic_input(batch_size)
                # Clone model parameters or zero grads
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(inputs)

                # Extract logits if dictionary or model output object
                if isinstance(outputs, dict):
                    outputs = outputs.get("logits", next(iter(outputs.values())))
                elif hasattr(outputs, "logits"):
                    outputs = outputs.logits
                elif isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                if self.target_shape is not None:
                    targets = torch.randn(batch_size, *self.target_shape, device=self.device)
                elif isinstance(self.loss_fn, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss, nn.HuberLoss)):
                    targets = torch.randn_like(outputs)
                elif isinstance(self.loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                    targets = torch.rand_like(outputs)
                elif outputs.ndim > 1:
                    num_classes = outputs.shape[-1]
                    targets = torch.randint(0, max(1, num_classes), (batch_size,), device=self.device)
                else:
                    targets = torch.zeros(batch_size, device=self.device)

                # Match loss computation
                if isinstance(self.loss_fn, nn.CrossEntropyLoss) and targets.dtype != torch.long:
                    targets = targets.long()

                loss = self.loss_fn(outputs, targets)
                loss.backward()

                # Synchronize if CUDA
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)

            peak_mem = tracker.peak_allocated
            success = True
            err_msg = None

            # Verify if peak memory exceeds resolved memory limit
            if self.device.type == "cuda" and peak_mem > self.resolved_memory_limit:
                success = False
                err_msg = f"Peak memory {format_bytes(peak_mem)} exceeded limit {format_bytes(self.resolved_memory_limit)}"

            return ProbeTrialResult(
                batch_size=batch_size,
                success=success,
                peak_memory_bytes=peak_mem,
                duration_seconds=tracker.duration_seconds,
                error_message=err_msg,
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
            err_str = str(e)
            is_oom = "out of memory" in err_str.lower() or "cuda oom" in err_str.lower()
            if "expected more than 1 value per channel" in err_str.lower():
                return ProbeTrialResult(
                    batch_size=batch_size,
                    success=False,
                    peak_memory_bytes=0,
                    duration_seconds=0.0,
                    error_message="BatchNorm requires batch_size > 1",
                )
            if not is_oom and "OutOfMemoryError" not in type(e).__name__:
                logger.warning(f"Trial encountered error: {err_str}")
                return ProbeTrialResult(
                    batch_size=batch_size,
                    success=False,
                    peak_memory_bytes=0,
                    duration_seconds=0.0,
                    error_message=err_str,
                )

            return ProbeTrialResult(
                batch_size=batch_size,
                success=False,
                peak_memory_bytes=self.resolved_memory_limit,
                duration_seconds=0.0,
                error_message="CUDA Out Of Memory",
            )
        finally:
            # Clean up all temporary tensors created during trial
            del inputs, targets, outputs, loss
            self.model.zero_grad(set_to_none=True)
            gc.collect()
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def search(self) -> BatchSearchResult:
        """Perform exponential search followed by binary search to locate optimal micro-batch size."""
        trials: List[ProbeTrialResult] = []
        low = self.min_batch_size
        high = self.max_batch_size

        # Phase 1: Exponential probing (powers of 2)
        curr = low
        best_working = None
        first_failure = None

        while curr <= high:
            trial = self._probe_trial(curr)
            trials.append(trial)
            if trial.success:
                best_working = curr
                curr *= 2
            else:
                first_failure = curr
                break

        if best_working is None:
            # Even min_batch_size failed or ran out of memory
            return BatchSearchResult(
                optimal_batch_size=self.min_batch_size,
                memory_limit_bytes=self.resolved_memory_limit,
                trials=trials,
                reason=f"Minimum batch size {self.min_batch_size} failed trial; using fallback {self.min_batch_size}.",
            )

        if first_failure is None:
            # Max batch size succeeded
            return BatchSearchResult(
                optimal_batch_size=best_working,
                memory_limit_bytes=self.resolved_memory_limit,
                trials=trials,
                reason=f"Successfully reached maximum search limit batch size {best_working}.",
            )

        # Phase 2: Binary search between best_working and first_failure
        b_low = best_working + 1
        b_high = first_failure - 1

        while b_low <= b_high:
            mid = (b_low + b_high) // 2
            trial = self._probe_trial(mid)
            trials.append(trial)
            if trial.success:
                best_working = mid
                b_low = mid + 1
            else:
                b_high = mid - 1

        return BatchSearchResult(
            optimal_batch_size=best_working,
            memory_limit_bytes=self.resolved_memory_limit,
            trials=trials,
            reason=f"Optimal batch size {best_working} determined with {format_bytes(self.resolved_memory_limit)} memory budget.",
        )
