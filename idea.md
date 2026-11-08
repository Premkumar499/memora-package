# Memora --- Project Idea

## 1. Overview

**Memora** is a proposed open-source Python framework that helps PyTorch
developers use limited GPU memory more effectively during model
training.

Its goal is to make common memory and training optimizations easier to
inspect, configure, and apply through a unified developer-facing API.

Memora is inspired by the practical value of tools such as Unsloth:
identify a painful workflow, make it easier and more efficient, and
expose the improvement through a usable framework. Memora is not an
Unsloth clone and will focus initially on general PyTorch training
rather than LLM fine-tuning.

> Working name: **Memora**. Check package-name, domain, and trademark
> availability before public release.

## 2. Problem

Training a deep-learning model can fail or become inefficient when GPU
memory is limited. Developers often have to manually:

-   Reduce batch size after CUDA out-of-memory errors.
-   Configure mixed precision and gradient scaling.
-   Implement gradient accumulation to preserve an effective batch size.
-   Add activation checkpointing to reduce activation memory.
-   Monitor allocated, reserved, and peak GPU memory.
-   Repeat experiments to find a stable configuration.

These tasks are repetitive, error-prone, and can require substantial
PyTorch and GPU knowledge.

## 3. Proposed solution

Memora will inspect the hardware and training workload, estimate or
benchmark memory requirements, and recommend or apply supported
optimizations.

The framework should be transparent: show what it changed, why it
changed it, the trade-offs, and how the user can override the decision.

### Example target API

``` python
from memora import Memora

trainer = Memora(
    model=model,
    dataset=dataset,
    batch_size="auto",
    precision="auto",
    gradient_accumulation="auto",
    activation_checkpointing="auto",
    gpu_memory_limit="auto",
)

trainer.fit(epochs=10)
```

This is a proposed API design, not an implemented interface. The final
API may change as the engineering design becomes clearer.

## 4. Target users

-   Students and independent ML/DL developers working with consumer
    GPUs.
-   Researchers experimenting with models on shared or constrained GPUs.
-   PyTorch developers who want clear memory diagnostics and
    optimization recommendations.
-   Teams that need repeatable training configurations across different
    GPU environments.

## 5. Initial scope

The first version will focus on standard PyTorch training and begin with
image-classification models.

### Core capabilities

1.  **Hardware discovery**
    -   Detect CUDA availability and GPU name.
    -   Report total, free, allocated, reserved, and peak memory where
        available.
    -   Detect relevant device capabilities.
2.  **Memory monitoring**
    -   Sample GPU memory during training.
    -   Track peak allocated and reserved memory.
    -   Record CPU RAM and training-step timing.
    -   Produce a structured run report.
3.  **Batch-size recommendation**
    -   Benchmark candidate micro-batch sizes using representative
        inputs.
    -   Find a configuration that fits within a user-defined memory
        budget.
    -   Use safe trial isolation and clean up temporary allocations.
    -   Explain that a successful probe is not a guarantee for every
        future batch.
4.  **Mixed precision**
    -   Support configurable FP32, FP16, and BF16 where hardware and
        model operations permit.
    -   Use appropriate autocast and gradient scaling behaviour.
    -   Never silently change precision when the user has explicitly
        disabled it.
5.  **Gradient accumulation**
    -   Calculate accumulation steps for a requested effective batch
        size.
    -   Handle loss scaling and optimizer-step boundaries correctly.
    -   Account for incomplete final batches and distributed-training
        limitations.
6.  **Activation checkpointing**
    -   Offer explicit integration for supported model regions.
    -   Explain the memory-versus-compute trade-off.
    -   Avoid blindly rewriting arbitrary model architectures.
7.  **Optimization report**
    -   Record selected settings and reasons.
    -   Report peak VRAM, throughput, step time, and failures.
    -   Compare baseline and optimized runs under equivalent conditions.

## 6. Non-goals for the first release

-   Replacing PyTorch or CUDA.
-   Guaranteeing that every model fits on every GPU.
-   Automatically modifying arbitrary model architectures without user
    consent.
-   Promising faster training in every workload; some memory-saving
    methods add compute overhead.
-   Full distributed training, multi-GPU scheduling, model parallelism,
    or LLM-specific kernels in the MVP.
-   Changing model accuracy or training semantics without reporting the
    trade-off.

## 7. Technical stack

-   Python 3.11 or newer, subject to dependency compatibility.
-   PyTorch and torchvision.
-   CUDA-enabled PyTorch for NVIDIA GPU testing.
-   `nvidia-ml-py` for NVIDIA hardware telemetry.
-   `psutil` for system RAM information.
-   `typer` and `rich` for a future CLI.
-   `pytest` for tests.
-   `ruff` for linting and formatting.
-   GitHub Actions for continuous integration.

Prefer PyTorch's own CUDA memory APIs for allocator statistics. NVML
reports device-level information; these measurements are related but not
interchangeable.

## 8. Proposed architecture

``` text
User PyTorch application
        |
        v
     Memora API
        |
        v
   Optimization Engine
        |
        +-- Hardware Analyzer
        +-- Memory Monitor
        +-- Workload Profiler
        +-- Batch-size Search
        +-- Precision Manager
        +-- Gradient Accumulation Manager
        +-- Checkpointing Integrations
        |
        v
 Training Configuration + Runtime Hooks
        |
        v
 PyTorch Training Loop
        |
        v
 Metrics, Logs, and Optimization Report
```

### Suggested package layout

``` text
memora/
├── memora/
│   ├── __init__.py
│   ├── engine.py
│   ├── config.py
│   ├── hardware/
│   │   ├── gpu_detector.py
│   │   └── memory_monitor.py
│   ├── optimizer/
│   │   ├── batch_optimizer.py
│   │   ├── precision_optimizer.py
│   │   ├── gradient_optimizer.py
│   │   └── checkpoint_optimizer.py
│   ├── profiler/
│   │   └── training_profiler.py
│   ├── reporting/
│   │   └── report.py
│   └── utils/
├── examples/
├── tests/
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

## 9. Differentiation and research questions

Memora should not simply combine existing PyTorch switches. Its value
should come from a reliable and transparent decision process.

Questions to investigate:

-   Can representative warm-up runs predict a safe micro-batch size?
-   How should Memora reserve a configurable VRAM safety margin?
-   How can it distinguish PyTorch allocator memory from other
    processes' GPU usage?
-   How can it choose between smaller batches, mixed precision, and
    checkpointing based on measured trade-offs?
-   How can it preserve effective batch size and optimizer semantics?
-   How can it avoid expensive or unsafe repeated OOM probes?
-   Can it produce reproducible recommendations across runs?

Before claiming novelty, compare the design and benchmarks with existing
tools such as PyTorch, Accelerate, DeepSpeed, PyTorch Lightning,
TorchAO, and other GPU-memory utilities.

## 10. Success criteria

The MVP is successful when:

-   It detects supported GPU hardware and reports memory accurately.
-   It monitors a training run without materially disrupting it.
-   It can benchmark candidate batch sizes safely for a supported
    example.
-   It can configure mixed precision and gradient accumulation
    correctly.
-   It can provide a useful report comparing baseline and optimized
    runs.
-   Its tests cover CPU-only operation and supported CUDA environments.
-   Benchmark results are reproducible and disclose hardware, software
    versions, model, input shape, and settings.

## 11. First demonstration

Use a small image-classification workload such as ResNet18 on a
reproducible dataset.

Demonstrate:

1.  Baseline PyTorch training with a fixed configuration.
2.  Memora hardware and memory analysis.
3.  Memora's recommended configuration.
4.  Training with the selected settings.
5.  Comparison of peak VRAM, throughput, step time, and validation
    accuracy.

Do not fabricate performance numbers. Publish only measurements
collected from actual runs.

## 12. Long-term direction

After a reliable general PyTorch MVP:

-   Add more CNN, audio, and transformer examples.
-   Add model-specific checkpointing integrations.
-   Explore optimizer-state memory optimization.
-   Explore inference memory optimization as a separate module.
-   Consider distributed and multi-GPU support.
-   Build a CLI and richer experiment reports.

The long-term vision is a practical, transparent memory-optimization
assistant for PyTorch developers.
