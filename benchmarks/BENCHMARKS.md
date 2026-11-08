# Memora Benchmark Results
- **Hardware**: CPU (x86_64, 8 cores)
- **CUDA Available**: False
- **PyTorch**: 2.14.0+cpu

| Workload | Baseline Batch | Memora Batch | Baseline Throughput | Optimized Throughput | Throughput Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MLP (Tabular/Dense) | 16 | 256 | 15335.85 smp/s | 127043.43 smp/s | 728.41% |
| CNN (Vision/Convolutional) | 8 | 128 | 1266.98 smp/s | 1024.8 smp/s | -19.11% |