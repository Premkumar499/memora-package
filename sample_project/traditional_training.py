"""Traditional PyTorch training: manual loop, fixed batch size, no memory tracking."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_risk_dataset
from model import RiskPredictionModel


def main():
    print("=== [Approach 1] Traditional Manual PyTorch Training ===")
    
    # 1. Device resolution (manual boilerplate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset & DataLoader (developer guesses a batch size, e.g. 16)
    train_dataset, val_dataset = get_risk_dataset()
    batch_size = 16  # Guessed manually
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model & Optimizer
    model = RiskPredictionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 4. Manual Training Loop (lots of boilerplate, no memory telemetry)
    start_time = time.time()
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    duration = time.time() - start_time
    print(f"Training completed in {duration:.4f} seconds.")
    print("Drawbacks:")
    print(" - Batch size 16 was a guess (might underutilize hardware or cause OOM).")
    print(" - No VRAM/RAM telemetry recorded.")
    print(" - No automatic mixed precision or gradient accumulation.")


if __name__ == "__main__":
    main()

