# Memora Cross-Computer Integration Guide 🌐

Memora is designed to be **hardware-agnostic and operating-system independent**. You can integrate it into any other computer (Linux, macOS, Windows, Cloud VMs, or Docker) with zero code rewrites.

---

## 🚀 Option 1: Install Pre-Built Wheel (Fastest on Any Machine)

You can copy the generated wheel file in `dist/` to any computer and install it with standard `pip`:

```bash
# On the target computer:
pip install memora-0.1.0-py3-none-any.whl
```

Or install directly from your Git repository:

```bash
pip install git+https://github.com/your-username/memora.git
```

---

## 🐳 Option 2: Docker / Container Integration (Zero Dependency Setup)

Memora includes a multi-platform `Dockerfile` and `docker-compose.yml`. This guarantees identical behavior across any operating system.

### Running with Docker Compose:
```bash
# 1. Start the container
docker compose up

# 2. Run the quickstart example inside the container
docker compose run quickstart
```

### For Computers with NVIDIA GPUs:
Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed, then uncomment the GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## 💻 Option 3: Seamless OS & Hardware Compatibility

When your code runs on different machines, you do **not** need to change any configuration. Memora's `device="auto"` automatically adapts to whatever hardware exists on that computer:

| Target Computer Hardware | Operating System | How Memora Adapts Automatically |
| :--- | :--- | :--- |
| **NVIDIA RTX 30/40 Series, A100, H100** | Linux / Windows | Selects native **BF16**, probes GPU VRAM, checks allocator stats. |
| **NVIDIA GTX 16/20 Series, T4, V100** | Linux / Windows | Selects **FP16** with automatic `torch.amp.GradScaler`. |
| **Apple Silicon (M1, M2, M3, M4)** | macOS | Selects **Apple MPS** accelerator with FP16 autocast. |
| **Any CPU Laptop or Cloud VM** | Linux, macOS, Windows | Safely uses multi-threaded **CPU** and FP32 (no CUDA crashes). |

---

## 🔍 Option 4: Verification on the New Computer (1-Command Smoke Test)

After copying the project or installing the package on a new machine, run the built-in deployment health check:

```bash
python scripts/deploy_check.py
```

It validates:
1. OS environment & Python version.
2. Hardware accelerator availability (CUDA, MPS, or CPU).
3. VRAM / RAM allocator status.
4. An end-to-end forward/backward optimization pass.

You can also run the CLI:
```bash
memora hardware
```

---

## 🔌 Option 5: Integrating with Existing PyTorch Code

To use Memora in an existing project on any computer, wrap your model and dataset with `Memora`:

```python
import torch
from memora import Memora

# Your existing model & dataset
my_model = ...
my_dataset = ...

# Memora automatically tunes memory for whatever computer it is running on
trainer = Memora(
    model=my_model,
    dataset=my_dataset,
    batch_size="auto",               # Finds safe batch size for THIS computer
    precision="auto",                # Chooses best precision for THIS GPU/CPU
    gradient_accumulation="auto",    # Preserves your desired effective batch size
    activation_checkpointing="auto", # Conserves VRAM on small GPUs
    device="auto",                   # CUDA on GPU rigs, MPS on Mac, CPU elsewhere
)

# Train and get the report
report = trainer.fit(epochs=5)
report.print_summary()
```

