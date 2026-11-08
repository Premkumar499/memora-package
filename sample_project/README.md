# Sample Project: Risk Classification with Memora 🚀

This sample project demonstrates how **Memora** transforms a standard PyTorch machine learning application.

---

## 📂 Project Structure

```text
sample_project/
├── model.py                  # PyTorch Deep Neural Network with Residual connections
├── dataset.py                # Financial transaction risk classification dataset
├── traditional_training.py   # Approach 1: Manual PyTorch training loop
├── memora_training.py        # Approach 2: Memora auto-optimized training
└── README.md
```

---

## 🥊 Side-by-Side Comparison

| Problem Faced by Developers | Traditional PyTorch | With Memora |
| :--- | :--- | :--- |
| **Batch Size Selection** | Trial-and-error guessing; frequent CUDA OOM crashes. | **`batch_size="auto"`** probes and selects the maximum safe batch size with isolated trials. |
| **Mixed Precision (AMP)** | Manual `torch.autocast`, manual `GradScaler`, checking compute capabilities. | **`precision="auto"`** automatically enables BF16 (Ampere+) or FP16 with GradScaler (Turing/Pascal). |
| **Effective Batch Size** | Complex math and manual `step()` / `zero_grad()` frequency counters. | **`gradient_accumulation="auto"`** handles loss scaling and optimizer step boundaries automatically. |
| **Memory Monitoring** | No visibility into peak VRAM or CPU RAM usage during training. | Detailed tracking of peak memory, allocator statistics, throughput, and step time. |
| **Portability across Computers** | Code breaks when moving between an RTX GPU server, an Apple Silicon Mac, and a CPU laptop. | Automatically adapts to **CUDA**, **MPS**, or **CPU** with zero code changes. |
| **Boilerplate Code** | ~50 lines of repetitive training loop code. | **4 lines of unified API code**. |

---

## 🏃 Run the Comparison

Activate your virtual environment, navigate to `sample_project`, and run both:

### 1. Run Traditional Approach:
```bash
python sample_project/traditional_training.py
```

### 2. Run Memora Optimized Approach:
```bash
python sample_project/memora_training.py
```

