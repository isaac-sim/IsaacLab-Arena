# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RelationSolver checkpoint configuration."""

import torch

import pytest

from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams


def test_default_checkpoints_are_capped_by_max_iters():
    """Default checkpoints include the maximum iteration cap."""

    params = RelationSolverParams(max_iters=120)

    assert params.get_checkpoints() == (25, 50, 100, 120)


def test_checkpoint_configuration_must_be_strictly_increasing():
    """Checkpoint iterations must form a strictly increasing sequence."""

    with pytest.raises(AssertionError, match="strictly increasing"):
        RelationSolverParams(checkpoint_iters=(25, 25, 100))


def test_position_history_defaults_off():
    """Position history capture is disabled by default."""

    assert RelationSolverParams().save_position_history is False


def test_bbox_device_is_cpu_even_when_cuda_is_available(monkeypatch):
    """BBOX solving remains on CPU even when CUDA is available."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert RelationSolver._select_device(mesh_collision_enabled=False) == torch.device("cpu")


def test_mesh_device_uses_cuda_when_available(monkeypatch):
    """MESH solving uses CUDA when it is available."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert RelationSolver._select_device(mesh_collision_enabled=True) == torch.device("cuda")
