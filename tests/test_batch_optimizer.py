"""Tests for automatic batch-size search."""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from memora.optimizer.batch_optimizer import BatchSearchResult, BatchSizeOptimizer, ProbeTrialResult


def test_batch_optimizer_search_mlp(mlp_model):
    optimizer = BatchSizeOptimizer(
        model=mlp_model,
        device=torch.device("cpu"),
        input_shape=(16,),
        min_batch_size=1,
        max_batch_size=32,
    )

    result = optimizer.search()
    assert isinstance(result, BatchSearchResult)
    assert result.optimal_batch_size >= 1
    assert result.optimal_batch_size <= 32
    assert len(result.trials) > 0
    assert result.trials[0].success is True


def test_batch_optimizer_batchnorm_minimum(conv_model):
    # Models with BatchNorm require batch_size >= 2
    optimizer = BatchSizeOptimizer(
        model=conv_model,
        device=torch.device("cpu"),
        input_shape=(3, 16, 16),
        min_batch_size=1,
        max_batch_size=16,
    )
    assert optimizer.min_batch_size >= 2

    result = optimizer.search()
    assert result.optimal_batch_size >= 2


def test_batch_optimizer_handles_oom(mlp_model):
    optimizer = BatchSizeOptimizer(
        model=mlp_model,
        device=torch.device("cpu"),
        input_shape=(16,),
        min_batch_size=2,
        max_batch_size=16,
    )

    # Mock _probe_trial to simulate OOM on batch sizes > 4
    orig_probe = optimizer._probe_trial

    def fake_probe(batch_size: int):
        if batch_size > 4:
            return ProbeTrialResult(
                batch_size=batch_size,
                success=False,
                peak_memory_bytes=0,
                duration_seconds=0.0,
                error_message="CUDA Out Of Memory",
            )
        return orig_probe(batch_size)

    with patch.object(optimizer, "_probe_trial", side_effect=fake_probe):
        result = optimizer.search()
        assert result.optimal_batch_size == 4
