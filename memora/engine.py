"""Core Memora training engine and unified developer API."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from memora.config import MemoraConfig, OptimizationRecommendation
from memora.hardware.gpu_detector import GPUDetector, HardwareReport
from memora.hardware.memory_monitor import MemoryMonitor
from memora.optimizer.batch_optimizer import BatchSearchResult, BatchSizeOptimizer
from memora.optimizer.checkpoint_optimizer import CheckpointOptimizer
from memora.optimizer.gradient_optimizer import GradientAccumulationOptimizer
from memora.optimizer.precision_optimizer import PrecisionOptimizer
from memora.profiler.training_profiler import TrainingProfiler
from memora.reporting.report import RunReport
from memora.utils.common import format_bytes, resolve_device, set_seed

logger = logging.getLogger("memora")


class Memora:
    """Unified memory-aware PyTorch trainer and optimizer.

    Inspects available hardware, searches for safe batch sizes within memory budgets,
    manages mixed-precision (FP16/BF16/FP32), configures gradient accumulation, and
    applies activation checkpointing transparently.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Optional[Dataset] = None,
        dataloader: Optional[DataLoader] = None,
        val_dataset: Optional[Dataset] = None,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        batch_size: Union[int, str] = "auto",
        effective_batch_size: Optional[int] = None,
        precision: Union[str, Any] = "auto",
        gradient_accumulation: Union[int, str] = "auto",
        activation_checkpointing: Union[bool, str] = "auto",
        gpu_memory_limit: Union[float, int, str] = "auto",
        safety_margin: float = 0.15,
        device: Union[str, torch.device] = "auto",
        seed: int = 42,
        report_output_dir: str = "./memora_reports",
    ):
        self.raw_model = model
        self.device = resolve_device(device)
        self.seed = seed
        set_seed(self.seed)

        # Store data sources
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        if self.dataset is None and self.dataloader is not None:
            self.dataset = self.dataloader.dataset

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.user_optimizer = optimizer
        self.report_output_dir = report_output_dir

        # Create typed configuration
        self.config = MemoraConfig(
            device=str(self.device),
            batch_size=batch_size,
            effective_batch_size=effective_batch_size,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation,
            activation_checkpointing=activation_checkpointing,
            gpu_memory_limit=gpu_memory_limit,
            safety_margin=safety_margin,
            report_output_dir=report_output_dir,
            seed=seed,
        )

        # Move model to resolved device
        self.model = self.raw_model.to(self.device)

        # Hardware inspection
        self.hardware_report = GPUDetector.get_hardware_report()

        # Engine internal state
        self._is_optimized = False
        self.recommendations: Optional[OptimizationRecommendation] = None
        self.batch_result: Optional[BatchSearchResult] = None
        self.precision_opt: Optional[PrecisionOptimizer] = None
        self.grad_accum_opt: Optional[GradientAccumulationOptimizer] = None
        self.checkpoint_opt: Optional[CheckpointOptimizer] = None
        self.profiler: Optional[TrainingProfiler] = None
        self.last_report: Optional[RunReport] = None

    def _infer_input_shape(self) -> Tuple[int, ...]:
        """Infer single-sample input shape from dataset or dataloader."""
        if self.dataset is not None and len(self.dataset) > 0:
            sample = self.dataset[0]
            if isinstance(sample, (tuple, list)):
                x = sample[0]
            elif isinstance(sample, dict):
                x = next(iter(sample.values()))
            else:
                x = sample
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)

        if self.dataloader is not None:
            for batch in self.dataloader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = next(iter(batch.values()))
                else:
                    x = batch
                if isinstance(x, torch.Tensor):
                    # Slice off batch dim
                    return tuple(x.shape[1:])
                break

        # Fallback default image shape (3, 32, 32)
        return (3, 32, 32)

    def optimize(self) -> OptimizationRecommendation:
        """Analyze workload and compute transparent optimization recommendations."""
        logger.info(f"Starting Memora optimization analysis on device '{self.device}'...")
        sample_shape = self._infer_input_shape()

        # 1. Activation Checkpointing
        self.checkpoint_opt = CheckpointOptimizer(
            model=self.model,
            enabled=self.config.activation_checkpointing,
            device=self.device,
        )
        self.model = self.checkpoint_opt.apply()

        # 2. Precision Optimization
        self.precision_opt = PrecisionOptimizer(
            device=self.device,
            mode=self.config.precision,
        )

        # 3. Batch Size Search
        if self.config.batch_size == "auto":
            max_b = self.config.max_batch_size
            if self.dataset is not None and len(self.dataset) > 0:
                max_b = min(max_b, len(self.dataset))

            batch_optimizer = BatchSizeOptimizer(
                model=self.model,
                device=self.device,
                input_shape=sample_shape,
                loss_fn=self.criterion,
                memory_limit=self.config.gpu_memory_limit,
                safety_margin=self.config.safety_margin,
                min_batch_size=self.config.min_batch_size,
                max_batch_size=max_b,
            )
            self.batch_result = batch_optimizer.search()
            resolved_batch_size = self.batch_result.optimal_batch_size
            probe_results = self.batch_result.to_dict()
        else:
            resolved_batch_size = int(self.config.batch_size)
            probe_results = {"manual_batch_size": resolved_batch_size}

        # 4. Gradient Accumulation
        self.grad_accum_opt = GradientAccumulationOptimizer(
            micro_batch_size=resolved_batch_size,
            effective_batch_size=self.config.effective_batch_size,
            accumulation_steps=self.config.gradient_accumulation_steps,
        )

        # 5. Compile Recommendations and Explanations
        dev_info = GPUDetector.get_device_info(self.device)
        reasons = {
            "batch_size": self.batch_result.reason if self.batch_result else f"User specified batch_size={resolved_batch_size}",
            "precision": self.precision_opt.reason,
            "gradient_accumulation": (
                f"Using {self.grad_accum_opt.accumulation_steps} accumulation step(s) "
                f"to yield effective batch size {self.grad_accum_opt.computed_effective_batch_size}."
            ),
            "checkpointing": self.checkpoint_opt.reason,
        }

        self.recommendations = OptimizationRecommendation(
            recommended_batch_size=resolved_batch_size,
            recommended_precision=self.precision_opt.selected_precision,
            recommended_grad_accum_steps=self.grad_accum_opt.accumulation_steps,
            recommended_checkpointing=self.checkpoint_opt.is_enabled,
            estimated_peak_memory_bytes=self.batch_result.trials[-1].peak_memory_bytes if self.batch_result and self.batch_result.trials else 0,
            memory_limit_bytes=self.batch_result.memory_limit_bytes if self.batch_result else dev_info.total_memory,
            reasons=reasons,
            probe_results=probe_results,
        )

        self._is_optimized = True
        return self.recommendations

    def _prepare_dataloaders(self, batch_size: int) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Instantiate training and validation DataLoaders with the chosen batch size."""
        # If model has BatchNorm layers and final batch might have 1 item, drop_last prevents BatchNorm crash
        has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in self.model.modules()
        )
        drop_last = False
        if has_bn and self.dataset is not None and len(self.dataset) % batch_size == 1 and len(self.dataset) > batch_size:
            drop_last = True

        if self.dataset is not None:
            train_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
            )
        elif self.dataloader is not None:
            # Recreate with chosen batch size
            train_loader = DataLoader(
                self.dataloader.dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
            )
        else:
            raise ValueError("No dataset or dataloader provided to Memora.")

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        elif self.val_dataloader is not None:
            val_loader = self.val_dataloader

        return train_loader, val_loader

    def fit(self, epochs: int = 1) -> RunReport:
        """Train the model with chosen optimizations and collect profiling metrics."""
        if not self._is_optimized:
            self.optimize()

        assert self.recommendations is not None
        assert self.precision_opt is not None
        assert self.grad_accum_opt is not None

        batch_size = self.recommendations.recommended_batch_size
        train_loader, val_loader = self._prepare_dataloaders(batch_size)

        # Setup optimizer
        if self.user_optimizer is not None:
            optimizer = self.user_optimizer
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.profiler = TrainingProfiler(device=self.device)
        self.model.train()

        total_steps = len(train_loader) * epochs
        global_step = 0
        final_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        logger.info(f"Starting training: {epochs} epoch(s), batch_size={batch_size}, total_steps={total_steps}")
        epoch_start_time = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            self.grad_accum_opt.reset()

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1
                self.profiler.start_step(global_step)

                # Unpack batch
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch["input"]
                    targets = batch["target"]
                else:
                    inputs = batch
                    targets = None

                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                # Forward pass with autocast
                with self.precision_opt.autocast_context():
                    outputs = self.model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]

                    if targets is not None:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = outputs.sum()

                    scaled_loss = self.grad_accum_opt.scale_loss(loss)

                self.profiler.mark_forward_done()

                # Backward pass
                self.precision_opt.backward(scaled_loss)

                # Optimizer step boundary
                is_accum_step = self.grad_accum_opt.should_step(batch_idx, len(train_loader))
                if is_accum_step:
                    self.precision_opt.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)

                self.profiler.end_step(batch_size=inputs.size(0))
                final_loss = float(loss.item())

                # Track simple accuracy if classification
                if targets is not None and outputs.ndim > 1:
                    preds = outputs.argmax(dim=-1)
                    correct_predictions += int((preds == targets).sum().item())
                    total_samples += inputs.size(0)

        total_training_duration = time.time() - epoch_start_time
        final_accuracy = round((correct_predictions / total_samples * 100), 2) if total_samples > 0 else None

        # Validation phase if available
        val_metrics: Dict[str, Any] = {}
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)

        # Assemble full RunReport
        run_id = f"memora_run_{int(time.time())}"
        profiler_summary = self.profiler.summary()

        report = RunReport(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hardware_info=self.hardware_report.to_dict(),
            config=self.config.to_dict(),
            optimization_decisions={
                "batch_size": batch_size,
                "precision": self.precision_opt.selected_precision,
                "uses_grad_scaler": self.precision_opt.use_scaler,
                "gradient_accumulation_steps": self.grad_accum_opt.accumulation_steps,
                "effective_batch_size": self.grad_accum_opt.computed_effective_batch_size,
                "activation_checkpointing": self.recommendations.recommended_checkpointing,
                "reasons": self.recommendations.reasons,
            },
            training_metrics={
                "epochs": epochs,
                "total_steps": global_step,
                "final_loss": round(final_loss, 4),
                "final_accuracy": final_accuracy,
                "total_training_seconds": round(total_training_duration, 3),
                "validation": val_metrics,
            },
            profiler_summary=profiler_summary,
            step_metrics=[m.to_dict() for m in self.profiler.step_metrics],
        )

        self.last_report = report

        # Save report if output directory specified
        if self.report_output_dir:
            os.makedirs(self.report_output_dir, exist_ok=True)
            report_file = os.path.join(self.report_output_dir, f"{run_id}.json")
            report.save_json(report_file)

        return report

    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on validation loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    x = batch.to(self.device)
                    y = None

                with self.precision_opt.autocast_context() if self.precision_opt else contextlib.nullcontext():
                    out = self.model(x)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if y is not None:
                        loss = self.criterion(out, y)
                        total_loss += float(loss.item()) * x.size(0)
                        if out.ndim > 1:
                            preds = out.argmax(dim=-1)
                            correct += int((preds == y).sum().item())
                        total += x.size(0)

        self.model.train()
        mean_loss = round(total_loss / total, 4) if total > 0 else 0.0
        accuracy = round((correct / total * 100), 2) if total > 0 else 0.0

        return {
            "validation_loss": mean_loss,
            "validation_accuracy": accuracy,
            "validation_samples": total,
        }

    def get_recommendations(self) -> Optional[OptimizationRecommendation]:
        """Return the calculated optimization recommendations."""
        if not self._is_optimized:
            self.optimize()
        return self.recommendations

    def get_report(self) -> Optional[RunReport]:
        """Return the latest RunReport."""
        return self.last_report
