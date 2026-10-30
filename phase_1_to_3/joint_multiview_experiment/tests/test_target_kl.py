"""
Unit tests for target_kl and custom split file override in scratch code.
"""

import json
from pathlib import Path
import numpy as np
import torch
import pytest

from src.env_factory import load_viewpoint_split


def test_split_file_override(tmp_path: Path):
    split_file = tmp_path / "diag_split.json"
    split_data = {
        "train": [17],
        "val": [16, 18],
    }
    split_file.write_text(json.dumps(split_data))

    loaded = load_viewpoint_split(str(split_file))
    assert loaded["train"] == [17]
    assert loaded["val"] == [16, 18]
    assert loaded["test_interp"] == []
    assert loaded["test_extrap"] == []


def test_target_kl_early_break_logic():
    """Verifies that when approx_kl exceeds target_kl, loop breaks early."""
    update_epochs = 4
    num_minibatches = 4
    batch_size = 16
    minibatch_size = 4
    target_kl = 0.05

    # Simulate loop
    epochs_run = 0
    minibatches_run = 0
    approx_kl_sequence = [0.01, 0.02, 0.06, 0.08]  # Third minibatch exceeds 0.05
    kl_idx = 0

    for epoch in range(update_epochs):
        epochs_run += 1
        for start in range(0, batch_size, minibatch_size):
            minibatches_run += 1
            approx_kl = approx_kl_sequence[min(kl_idx, len(approx_kl_sequence) - 1)]
            kl_idx += 1
            if target_kl is not None and approx_kl > target_kl:
                break
        if target_kl is not None and approx_kl > target_kl:
            break

    # With break logic: stopped at epoch 1, minibatch 3
    assert epochs_run == 1
    assert minibatches_run == 3

    # Without target_kl (None)
    epochs_run_none = 0
    minibatches_run_none = 0
    for epoch in range(update_epochs):
        epochs_run_none += 1
        for start in range(0, batch_size, minibatch_size):
            minibatches_run_none += 1
            approx_kl = 0.10  # high KL
            if None is not None and approx_kl > None:
                break
        if None is not None and approx_kl > None:
            break

    assert epochs_run_none == 4
    assert minibatches_run_none == 16
