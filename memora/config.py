"""Configuration models and schemas for Memora."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


PrecisionType = Literal["auto", "fp32", "fp16", "bf16"]


class MemoraConfig(BaseModel):
    """Configuration for Memora optimization and training run."""

    # Device & resource management
    device: str = Field(
        default="auto",
        description="Target compute device: 'auto', 'cuda', 'cuda:0', 'mps', or 'cpu'.",
    )
    gpu_memory_limit: Union[float, int, Literal["auto"]] = Field(
        default="auto",
        description="VRAM memory limit. If float <= 1.0, fraction of total VRAM. If int > 1, bytes limit. 'auto' uses default safety headroom.",
    )
    safety_margin: float = Field(
        default=0.15,
        ge=0.01,
        le=0.50,
        description="Fraction of memory reserved as a safety buffer against unexpected memory spikes (default: 15%).",
    )

    # Batch search parameters
    batch_size: Union[int, Literal["auto"]] = Field(
        default="auto",
        description="Micro-batch size for training. 'auto' benchmarks and selects optimal size.",
    )
    min_batch_size: int = Field(
        default=1,
        ge=1,
        description="Minimum batch size allowed during search.",
    )
    max_batch_size: int = Field(
        default=512,
        ge=1,
        description="Maximum batch size allowed during search.",
    )

    # Gradient accumulation parameters
    effective_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target effective batch size. When provided with small batch_size, gradient accumulation is automatically configured.",
    )
    gradient_accumulation_steps: Union[int, Literal["auto"]] = Field(
        default="auto",
        description="Gradient accumulation steps. 'auto' computes steps to match effective_batch_size or 1.",
    )

    # Precision
    precision: PrecisionType = Field(
        default="auto",
        description="Numerical precision: 'auto', 'fp32', 'fp16', or 'bf16'.",
    )

    # Activation checkpointing
    activation_checkpointing: Union[bool, Literal["auto"]] = Field(
        default="auto",
        description="Activation checkpointing toggle: 'auto' enables it when VRAM is tight, or explicit boolean.",
    )

    # Profiling & reporting
    enable_profiling: bool = Field(
        default=True,
        description="Whether to monitor and profile memory and step timings.",
    )
    report_output_dir: Optional[str] = Field(
        default="./memora_reports",
        description="Directory to persist optimization reports.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic runs.",
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v.lower() != "auto":
                raise ValueError("batch_size string value must be 'auto'")
            return "auto"
        if isinstance(v, int):
            if v < 1:
                raise ValueError("batch_size must be >= 1")
            return v
        raise ValueError(f"Invalid batch_size type: {type(v)}")

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_grad_accum(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v.lower() != "auto":
                raise ValueError("gradient_accumulation_steps string must be 'auto'")
            return "auto"
        if isinstance(v, int):
            if v < 1:
                raise ValueError("gradient_accumulation_steps must be >= 1")
            return v
        raise ValueError(f"Invalid gradient_accumulation_steps type: {type(v)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()


class OptimizationRecommendation(BaseModel):
    """Structured recommendation produced by the optimization engine."""

    recommended_batch_size: int
    recommended_precision: Literal["fp32", "fp16", "bf16"]
    recommended_grad_accum_steps: int
    recommended_checkpointing: bool
    estimated_peak_memory_bytes: int
    memory_limit_bytes: int
    reasons: Dict[str, str] = Field(default_factory=dict)
    probe_results: Dict[str, Any] = Field(default_factory=dict)

