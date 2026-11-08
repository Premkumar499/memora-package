"""End-to-end tests for Memora trainer engine."""

import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from memora import Memora


def test_memora_engine_full_lifecycle(mlp_model, synthetic_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Memora(
            model=mlp_model,
            dataset=synthetic_dataset,
            batch_size="auto",
            precision="auto",
            gradient_accumulation="auto",
            activation_checkpointing="auto",
            device="cpu",
            report_output_dir=tmpdir,
        )

        # 1. Optimize preview
        recs = trainer.optimize()
        assert recs.recommended_batch_size >= 1
        assert recs.recommended_precision == "fp32"
        assert recs.recommended_grad_accum_steps >= 1

        # 2. Fit training
        report = trainer.fit(epochs=2)
        assert report.run_id != ""
        assert report.training_metrics["epochs"] == 2
        assert report.training_metrics["total_steps"] > 0
        assert report.profiler_summary["total_steps"] > 0


def test_memora_with_validation_set(mlp_model, synthetic_dataset):
    val_x = torch.randn(16, 16)
    val_y = torch.randint(0, 4, (16,))
    val_dataset = TensorDataset(val_x, val_y)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Memora(
            model=mlp_model,
            dataset=synthetic_dataset,
            val_dataset=val_dataset,
            batch_size=8,
            device="cpu",
            report_output_dir=tmpdir,
        )

        report = trainer.fit(epochs=1)
        assert "validation" in report.training_metrics
        assert "validation_loss" in report.training_metrics["validation"]


def test_memora_with_custom_optimizer_and_loss(mlp_model, synthetic_dataset):
    custom_opt = torch.optim.SGD(mlp_model.parameters(), lr=0.05, momentum=0.9)
    custom_loss = nn.CrossEntropyLoss()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Memora(
            model=mlp_model,
            dataset=synthetic_dataset,
            optimizer=custom_opt,
            criterion=custom_loss,
            batch_size=16,
            device="cpu",
            report_output_dir=tmpdir,
        )
        report = trainer.fit(epochs=1)
        assert report.training_metrics["total_steps"] > 0


def test_memora_regression_mse_loss():
    # Test regression with MSELoss
    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 1)

        def forward(self, x):
            return self.linear(x)

    model = Regressor()
    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    dataset = TensorDataset(x, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Memora(
            model=model,
            dataset=dataset,
            criterion=nn.MSELoss(),
            batch_size="auto",
            device="cpu",
            report_output_dir=tmpdir,
        )
        report = trainer.fit(epochs=1)
        assert report.training_metrics["total_steps"] > 0
        assert report.training_metrics["final_loss"] >= 0.0


def test_memora_dict_batch_support():
    # Custom dataset yielding dictionary
    class DictDataset:
        def __init__(self):
            self.data = [{"features": torch.randn(8), "labels": torch.randint(0, 2, (1,)).squeeze()} for _ in range(32)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    model = nn.Linear(8, 2)
    dataset = DictDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Memora(
            model=model,
            dataset=dataset,
            batch_size=8,
            device="cpu",
            report_output_dir=tmpdir,
        )
        report = trainer.fit(epochs=1)
        assert report.training_metrics["total_steps"] > 0


def test_memora_empty_dataloader_raises():
    import pytest
    empty_dataset = TensorDataset(torch.empty(0, 8), torch.empty(0))
    trainer = Memora(
        model=nn.Linear(8, 2),
        dataset=empty_dataset,
        batch_size=4,
        device="cpu",
    )
    with pytest.raises(ValueError, match="Dataset contains 0 samples|DataLoader contains 0 batches"):
        trainer.fit(epochs=1)


