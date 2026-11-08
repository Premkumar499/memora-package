# Memora 🧠⚡

> **A memory-aware training optimization framework for PyTorch across any OS & hardware.**

Memora helps PyTorch developers use limited GPU memory (and CPU resources) more effectively during model training. It inspects hardware, benchmarks memory requirements safely, and automatically configures batch sizes, mixed precision, gradient accumulation, and activation checkpointing through a single unified developer API.

---

## 🚀 Key Features

- **🌐 Works on Any OS & System**: Seamlessly runs on Linux, macOS (Apple Silicon MPS), and Windows. Detects CUDA GPUs (e.g. RTX 3050, T4, A100), Apple MPS, and CPU with graceful fallbacks.
- **🔍 Safe Micro-Batch Search**: Probes candidate batch sizes with isolated trial runs, catching CUDA Out-Of-Memory (OOM) errors and enforcing configurable memory safety margins without crashing your workflow.
- **⚡ Adaptive Mixed Precision (AMP)**: Automatically selects BF16 on modern Ampere+ GPUs (Compute Capability >= 8.0) for zero-scaler speed, FP16 with `torch.amp.GradScaler` on Volta/Turing/Pascal GPUs, or FP32 on CPU.
- **⚖️ Gradient Accumulation**: Preserves a target effective batch size regardless of hardware VRAM constraints by automatically computing micro-batches and accumulation steps with normalized loss gradients.
- **💾 Activation Checkpointing**: Automatically advises or wraps layers with `torch.utils.checkpoint` to conserve 40–60% activation memory when training under constrained VRAM.
- **📊 Transparent Profiling & Reporting**: Generates structured JSON and Markdown execution reports comparing baseline PyTorch vs. optimized configurations (peak memory, step times, throughput, accuracy).
- **🖥️ Rich CLI**: Instant hardware discovery and report viewer via `memora hardware` and `memora report`.

---

## 📦 Installation

Memora requires Python 3.9+ and PyTorch >= 2.0.

```bash
# Clone the repository
git clone https://github.com/Premkumar499/memora-package.git
cd memora-package

# Install in editable mode
pip install -e .
```

---

## 🛠️ Quickstart

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from memora import Memora

# 1. Define your PyTorch model and dataset
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

dataset = TensorDataset(
    torch.randn(256, 64),
    torch.randint(0, 10, (256,)),
)

# 2. Instantiate Memora with automatic memory optimization
trainer = Memora(
    model=model,
    dataset=dataset,
    batch_size="auto",               # Automatically find optimal batch size
    precision="auto",                # Automatically select BF16/FP16/FP32
    gradient_accumulation="auto",    # Compute accumulation steps
    activation_checkpointing="auto", # Enable checkpointing if VRAM is tight
    device="auto",                   # 'cuda', 'mps', or 'cpu'
)

# 3. Preview recommendations (optional)
trainer.optimize()

# 4. Train model and collect performance report
report = trainer.fit(epochs=3)

# 5. Display execution report
report.print_summary()
```

---

## 💻 Command Line Interface (CLI)

Memora includes a built-in CLI powered by Typer and Rich:

### 1. Hardware Discovery
Inspect available compute devices, memory limits, and compute capabilities:
```bash
memora hardware
```

### 2. View Training Run Reports
Pretty-print detailed memory and throughput metrics from a saved run JSON:
```bash
memora report <path-to-report.json>
```

### 3. Version Check
```bash
memora version
```

---

## 📁 Repository Structure

```text
framework/
├── memora/
│   ├── __init__.py           # Public exports
│   ├── config.py             # Pydantic schemas & configuration
│   ├── engine.py             # Core Memora trainer API
│   ├── cli.py                # Rich Typer CLI
│   ├── hardware/
│   │   ├── gpu_detector.py   # CUDA, MPS, CPU & NVML telemetry
│   │   └── memory_monitor.py # Peak memory tracking & snapshots
│   ├── optimizer/
│   │   ├── batch_optimizer.py      # Exponential & binary batch search
│   │   ├── precision_optimizer.py  # Autocast & GradScaler manager
│   │   ├── gradient_optimizer.py   # Effective batch accumulation
│   │   └── checkpoint_optimizer.py # Activation checkpointing
│   ├── profiler/
│   │   └── training_profiler.py    # Step timings & throughput
│   ├── reporting/
│   │   └── report.py         # RunReport & baseline comparison
│   └── utils/
│       └── common.py         # Formatting, device resolution & seeds
├── scripts/
│   └── deploy_check.py       # Deployment health check smoke test
├── docs/
│   └── INTEGRATION_GUIDE.md  # Cross-computer integration guide
├── pyproject.toml            # PEP 517/621 packaging metadata
├── README.md                 # Project documentation
├── LICENSE                   # Apache 2.0 license
├── idea.md                   # Concept & architectural motivation
└── plan.md                   # Development milestones & roadmap
```

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.

