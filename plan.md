# Memora --- Development Plan

## 1. Project objective

Build an open-source Python framework that analyzes PyTorch training
workloads and helps developers choose memory-aware training
configurations.

**Initial target:** single-GPU PyTorch training, beginning with
CNN/image-classification workloads.

**Primary development hardware:** NVIDIA RTX 3050 6GB when available.
Use Kaggle NVIDIA T4 for additional validation. Keep CPU-only tests so
development does not depend on a GPU.

## 2. Development principles

1.  Build and test one small module at a time.
2.  Keep optimization decisions measurable and explainable.
3.  Preserve training semantics wherever possible.
4.  Never promise a configuration will fit solely from a theoretical
    estimate; validate with representative workloads.
5.  Make automatic behaviour opt-in and provide manual overrides.
6.  Benchmark against a clearly defined PyTorch baseline.
7.  Treat unsupported models and hardware gracefully.

## 3. Milestones

### Phase 0 --- Research and design

**Tasks** - Review PyTorch CUDA memory APIs, AMP, gradient accumulation,
and activation checkpointing. - Study NVML and its distinction from
PyTorch allocator statistics. - Compare PyTorch, Accelerate, DeepSpeed,
PyTorch Lightning, TorchAO, and related tools. - Define supported
Python/PyTorch/CUDA versions. - Define a small supported training
interface and configuration schema.

**Deliverables** - Architecture diagram. - Supported-feature matrix. -
Baseline benchmark specification. - Initial README and project licence
decision.

**Exit criteria** - Every MVP feature has a defined behaviour,
limitation, and test strategy.

### Phase 1 --- Project foundation and hardware discovery

**Tasks** - Create the Python package and virtual environment. - Add
`pyproject.toml`, package metadata, linting, and tests. - Implement CUDA
availability detection. - Detect GPU name, total memory, and device
capability. - Implement CPU-only fallback. - Add structured hardware
information output.

**Deliverables** - Installable local package. -
`hardware/gpu_detector.py`. - Unit tests for CUDA-present and
CUDA-absent behaviour. - Hardware information example.

**Exit criteria** - Hardware detection works without crashing on
CPU-only systems. - Missing NVML or unavailable CUDA produces clear
messages.

### Phase 2 --- Memory monitoring

**Tasks** - Read PyTorch allocated and reserved CUDA memory. - Track
peak allocated and reserved memory. - Add optional NVML device-level
monitoring. - Record CPU RAM and step duration. - Implement start,
sample, and stop monitoring lifecycle. - Save measurements in JSON or
CSV.

**Deliverables** - `hardware/memory_monitor.py`. -
`profiler/training_profiler.py`. - Sample monitoring script. - Tests
using mocks for hardware-dependent calls.

**Exit criteria** - Monitoring produces a report for a sample PyTorch
training run. - Reports distinguish process allocator metrics from
whole-device metrics.

### Phase 3 --- Automatic batch-size search

**Tasks** - Define a supported probe interface for model, sample input,
loss, and forward/backward operation. - Create isolated candidate
trials. - Start with a conservative candidate and search for a safe
micro-batch size. - Catch CUDA OOM errors without hiding unrelated
exceptions. - Clean up temporary tensors and gradients between trials. -
Add configurable memory safety margin. - Record trial outcomes and
timings. - Document limitations for dynamic shapes, random data,
stateful models, and variable-length inputs.

**Deliverables** - `optimizer/batch_optimizer.py`. - Batch-size search
example. - Tests for success, OOM, and non-OOM exceptions.

**Exit criteria** - Search finds a working candidate for supported
deterministic example workloads. - A failed candidate does not leave the
next trial in a corrupted state.

### Phase 4 --- Mixed precision

**Tasks** - Add precision modes: `fp32`, `fp16`, `bf16`, and `auto`. -
Check device capability and user constraints. - Integrate
`torch.autocast`. - Integrate `torch.amp.GradScaler` for FP16 where
appropriate. - Handle BF16 without unnecessary gradient scaling. -
Document numerical stability and unsupported operations.

**Deliverables** - `optimizer/precision_optimizer.py`. - Precision
examples. - Tests for configuration selection and training integration.

**Exit criteria** - Supported precision modes complete a sample training
step. - The chosen precision and scaling behaviour appear in the run
report.

### Phase 5 --- Gradient accumulation

**Tasks** - Accept a requested effective batch size. - Calculate
micro-batch and accumulation steps. - Normalize loss correctly across
accumulation steps. - Handle optimizer zeroing and stepping at correct
boundaries. - Handle incomplete final accumulation groups. - Document
interactions with schedulers, gradient clipping, BatchNorm, and
distributed training.

**Deliverables** - `optimizer/gradient_optimizer.py`. -
Effective-batch-size example. - Tests comparing update behaviour on a
controlled workload.

**Exit criteria** - Accumulation behaves as documented and reports the
effective batch size accurately.

### Phase 6 --- Activation checkpointing

**Tasks** - Design explicit checkpointing configuration. - Integrate
PyTorch checkpoint utilities for supported model regions. - Test forward
and backward correctness. - Measure memory and step-time trade-offs. -
Provide clear warnings for unsupported or unsafe integrations.

**Deliverables** - `optimizer/checkpoint_optimizer.py`. - Example model
integration. - Tests for gradient correctness.

**Exit criteria** - A supported model trains with checkpointing enabled
and reports the observed trade-off.

### Phase 7 --- Optimization engine and public API

**Tasks** - Define a stable configuration object. - Integrate hardware
analysis, profiling, batch search, precision, and accumulation. - Build
a recommendation mode that does not modify training. - Build an opt-in
apply mode. - Provide manual overrides for each optimization. - Add a
transparent explanation for every recommendation. - Define the first
supported training-loop integration.

**Deliverables** - `engine.py`. - `config.py`. - Public `Memora` API. -
End-to-end example.

**Exit criteria** - A developer can run the supported example with a
simple API and inspect every selected setting.

### Phase 8 --- Reporting, CLI, and documentation

**Tasks** - Create structured run reports. - Include GPU details,
settings, peak VRAM, step time, throughput, and validation metrics where
available. - Add a basic Typer CLI for hardware inspection and report
viewing. - Write installation, quick-start, API, limitations, and
troubleshooting documentation. - Add examples and version compatibility
guidance.

**Deliverables** - `reporting/report.py`. - `cli.py`. - README and
documentation. - Reproducible examples.

**Exit criteria** - A new user can install the package, run an example,
and understand its report.

### Phase 9 --- Benchmarking and release

**Tasks** - Define baseline and optimized runs with identical model,
data, and evaluation conditions. - Benchmark ResNet18 first, then
ResNet50 or another supported CNN. - Measure peak VRAM, throughput, step
time, total training time, and validation quality. - Repeat runs and
report variability. - Test on RTX 3050 and Kaggle T4 when available. -
Add GitHub Actions for CPU tests and supported static checks. - Prepare
package metadata, licence, changelog, and release.

**Deliverables** - Reproducible benchmark scripts. - Benchmark report
with actual measurements. - GitHub repository. - First versioned package
release when quality criteria are met.

**Exit criteria** - Results are repeatable, limitations are documented,
and no performance claims rely on illustrative numbers.

## 4. Suggested implementation order

Build in this order to reduce complexity:

1.  Package skeleton and tests.
2.  GPU detector.
3.  Memory monitor.
4.  Fixed-configuration baseline training example.
5.  Batch-size probe for the example model.
6.  Mixed precision.
7.  Gradient accumulation.
8.  Activation checkpointing.
9.  Integrated recommendation engine.
10. Reports, CLI, documentation, and benchmarks.

Do not begin with a fully automatic training engine. First prove each
optimization independently.

## 5. Initial repository structure

``` text
memora/
├── memora/
│   ├── __init__.py
│   ├── engine.py
│   ├── config.py
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── gpu_detector.py
│   │   └── memory_monitor.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── batch_optimizer.py
│   │   ├── precision_optimizer.py
│   │   ├── gradient_optimizer.py
│   │   └── checkpoint_optimizer.py
│   ├── profiler/
│   │   ├── __init__.py
│   │   └── training_profiler.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report.py
│   └── utils/
├── examples/
│   └── train_resnet.py
├── tests/
│   ├── test_gpu_detector.py
│   ├── test_memory_monitor.py
│   └── test_optimizer.py
├── docs/
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## 6. Benchmark protocol

For each supported workload:

-   Record GPU model, VRAM, driver, CUDA, Python, and PyTorch versions.
-   Record model architecture, input dimensions, dataset, and number of
    samples.
-   Fix random seeds where practical.
-   Use the same model initialization and data conditions for
    comparisons.
-   Separate warm-up from measured iterations.
-   Synchronize CUDA around timing measurements.
-   Measure peak allocated and reserved memory and explain which metric
    is reported.
-   Measure samples per second and step time.
-   Compare validation metrics to detect changes in training behaviour.
-   Repeat trials and report mean and variability.
-   Disclose OOM trials and failed configurations.

Compare: 1. Baseline PyTorch. 2. Manually optimized PyTorch. 3. Memora
recommendations and applied configuration.

Do not assume Memora is faster. Some memory-saving techniques can reduce
throughput.

## 7. Risks and mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  OOM during probing                  Isolated trials, cleanup,
                                      conservative limits

  Other processes use VRAM            Combine process and device-level
                                      monitoring; disclose uncertainty

  Probe does not represent real       Support representative inputs and
  batches                             dynamic-shape warnings

  Mixed precision changes numerical   Explicit modes, validation, and
  behaviour                           opt-out

  Accumulation changes training       Correct loss normalization and
  semantics                           documented limitations

  Checkpointing slows training        Make optional and report trade-offs

  API becomes too broad               Keep MVP limited to single-GPU
                                      PyTorch workloads

  Existing tools already cover        Benchmark and identify a concrete
  features                            usability or automation gap
  -----------------------------------------------------------------------

## 8. Definition of MVP complete

The MVP is ready for an initial public release when:

-   [ ] Package installs in a clean environment.
-   [ ] GPU and CPU hardware detection work.
-   [ ] Memory monitoring creates a useful report.
-   [ ] Batch-size search works on the supported example.
-   [ ] Mixed precision works on supported hardware.
-   [ ] Gradient accumulation preserves the documented effective batch
    size.
-   [ ] Checkpointing works for at least one documented model
    integration.
-   [ ] Automatic recommendations can be inspected and overridden.
-   [ ] Tests pass in supported environments.
-   [ ] Baseline comparisons use real, reproducible measurements.
-   [ ] Documentation explains limitations and compatibility.

## 9. Immediate next task

Start with **Phase 1: Project foundation and hardware discovery**.

1.  Create the repository and virtual environment.
2.  Install PyTorch and the hardware-monitoring dependencies.
3.  Implement `gpu_detector.py`.
4.  Test CUDA detection and CPU fallback.
5.  Print a clean hardware report.

Only after Phase 1 works should development move to memory monitoring.
