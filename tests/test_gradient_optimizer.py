"""Tests for gradient accumulation optimizer."""

import pytest
import torch

from memora.optimizer.gradient_optimizer import GradientAccumulationOptimizer


def test_effective_batch_size_calculation():
    # Micro-batch 4, target effective batch 16 -> 4 steps
    opt = GradientAccumulationOptimizer(micro_batch_size=4, effective_batch_size=16)
    assert opt.accumulation_steps == 4
    assert opt.computed_effective_batch_size == 16

    # Micro-batch 5, target effective batch 16 -> 4 steps (ceil)
    opt2 = GradientAccumulationOptimizer(micro_batch_size=5, effective_batch_size=16)
    assert opt2.accumulation_steps == 4
    assert opt2.computed_effective_batch_size == 20


def test_loss_scaling():
    opt = GradientAccumulationOptimizer(micro_batch_size=4, accumulation_steps=4)
    loss = torch.tensor(8.0)
    scaled = opt.scale_loss(loss)
    assert scaled.item() == 2.0


def test_should_step_boundaries():
    opt = GradientAccumulationOptimizer(micro_batch_size=4, accumulation_steps=3)
    total_batches = 8

    # Steps 0, 1 should NOT step
    assert opt.should_step(0, total_batches) is False
    assert opt.should_step(1, total_batches) is False

    # Step 2 (3rd batch) SHOULD step
    assert opt.should_step(2, total_batches) is True

    # Step 3, 4 should NOT step
    assert opt.should_step(3, total_batches) is False
    assert opt.should_step(4, total_batches) is False

    # Step 5 (6th batch) SHOULD step
    assert opt.should_step(5, total_batches) is True

    # Step 6 should NOT step
    assert opt.should_step(6, total_batches) is False

    # Step 7 (8th batch, final batch of epoch) SHOULD step to flush remaining gradients
    assert opt.should_step(7, total_batches) is True

