# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuroboIKSolver configclass-safety, no GPU or cuRobo build needed.

The solver rides in the placement pool Isaac Lab stores in an ``EventTermCfg``, whose configclass both
deep-copies the params and recursively walks every ``__dict__`` with no cycle guard. The wrapped cuRobo
``IKSolver`` (un-picklable CUDA/ctypes handles, deep cyclic graph) breaks both passes, so the solver shares
itself across copies (``__deepcopy__``) and exposes no ``__dict__`` (``__slots__``). A ctypes pointer stands
in for the wrapped solver, with ``__init__`` bypassed.
"""

from __future__ import annotations

import copy
import ctypes

import pytest


@pytest.mark.curobo_deps
def test_solver_is_shared_across_deepcopy():
    """Deep-copying a solver (even nested in a container) returns the same live instance."""
    from isaaclab_arena_curobo.ik_solver import CuroboIKSolver

    solver = object.__new__(CuroboIKSolver)  # bypass __init__: no GPU / cuRobo build
    # A ctypes pointer stands in for the cuRobo IKSolver's un-copyable CUDA/ctypes handles.
    solver.ik_solver = ctypes.pointer(ctypes.c_int(1))

    # Nested like the real EventTermCfg params (pool -> placer -> validator -> solver).
    params = {"placement_pool": {"placer": {"solver": solver}}, "objects": [1, 2]}
    copied = copy.deepcopy(params)

    assert copied["placement_pool"]["placer"]["solver"] is solver
    assert copied["objects"] == [1, 2] and copied["objects"] is not params["objects"]


@pytest.mark.curobo_deps
def test_solver_has_no_dict_so_configclass_validate_cannot_recurse():
    """The solver exposes no ``__dict__``, the one thing that stops Isaac Lab's cycle-guardless
    ``configclass._validate`` from descending into the wrapped cuRobo solver's deep, cyclic graph."""
    from isaaclab_arena_curobo.ik_solver import CuroboIKSolver

    solver = object.__new__(CuroboIKSolver)  # bypass __init__: no GPU / cuRobo build
    solver.ik_solver = ctypes.pointer(ctypes.c_int(1))

    assert not hasattr(solver, "__dict__")
