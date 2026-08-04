# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil

# Packages whose task cfgs must mark their time-out term as a truncation.
TASK_PACKAGES = ("isaaclab_arena.tasks", "isaaclab_arena_examples.external_environments")


def _iter_time_out_terms():
    """Yield ``(module_name, cls_name, term)`` for every configclass exposing a ``time_out`` field."""
    from isaaclab.managers import TerminationTermCfg

    for package_name in TASK_PACKAGES:
        package = importlib.import_module(package_name)
        module_names = [name for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}.")]
        for module_name in [package_name, *module_names]:
            module = importlib.import_module(module_name)
            for cls in vars(module).values():
                # Only classes defined in this module, so a cfg imported elsewhere is not double-counted.
                if not isinstance(cls, type) or getattr(cls, "__module__", None) != module_name:
                    continue
                if "time_out" not in getattr(cls, "__annotations__", {}):
                    continue
                term = cls().time_out
                if isinstance(term, TerminationTermCfg):
                    yield module_name, cls.__name__, term


def test_all_task_cfgs_mark_time_out_as_truncation():
    """``TerminationTermCfg.time_out`` defaults to False, which makes ``TerminationManager``
    report episode-length expiry as ``terminated`` instead of ``truncated``.

    Walks the task packages rather than listing cfgs, so a new task that forgets the flag
    fails here without anyone remembering to update this test.
    """
    terms = list(_iter_time_out_terms())
    assert terms, f"No time_out terms discovered in {TASK_PACKAGES}; the sweep is not testing anything."

    offenders = [f"{module}.{cls_name}" for module, cls_name, term in terms if term.time_out is not True]
    assert not offenders, f"Termination cfgs declaring a time_out term without time_out=True: {offenders}"
