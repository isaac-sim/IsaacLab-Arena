# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_add_suffix_configclass_transform(simulation_app) -> bool:
    """Test that _add_suffix_configclass_transform correctly renames fields with suffix."""

    from functools import partial

    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.utils.configclass import transform_configclass_instance

    @configclass
    class FooCfg:
        int_field: int = 123
        str_field: str = "123"
        float_field: float = 1.23
        bool_field: bool = True

    try:
        original_cfg = FooCfg()
        edited_cfg = transform_configclass_instance(
            original_cfg,
            partial(CompositeTaskBase._add_suffix_configclass_transform, suffix="_suffix"),
        )

        # Check that new fields exist with suffix
        assert hasattr(edited_cfg, "int_field_suffix")
        assert hasattr(edited_cfg, "str_field_suffix")
        assert hasattr(edited_cfg, "float_field_suffix")
        assert hasattr(edited_cfg, "bool_field_suffix")

        # Check that values are preserved
        assert edited_cfg.int_field_suffix == 123
        assert edited_cfg.str_field_suffix == "123"
        assert edited_cfg.float_field_suffix == 1.23
        assert edited_cfg.bool_field_suffix is True

        # Check types are preserved
        assert isinstance(edited_cfg.int_field_suffix, int)
        assert isinstance(edited_cfg.str_field_suffix, str)
        assert isinstance(edited_cfg.float_field_suffix, float)
        assert isinstance(edited_cfg.bool_field_suffix, bool)

        # Check that old field names don't exist
        assert not hasattr(edited_cfg, "int_field")
        assert not hasattr(edited_cfg, "str_field")
        assert not hasattr(edited_cfg, "float_field")
        assert not hasattr(edited_cfg, "bool_field")

        # Test None input
        edited_cfg = transform_configclass_instance(
            None,
            partial(CompositeTaskBase._add_suffix_configclass_transform, suffix="_suffix"),
        )
        assert edited_cfg is None

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_remove_configclass_transform(simulation_app) -> bool:
    """Test that _remove_configclass_transform correctly removes specified fields."""

    from functools import partial

    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.utils.configclass import transform_configclass_instance

    @configclass
    class FooCfg:
        field_a: int = 123
        field_b: str = "123"
        field_c: float = 1.23

    try:
        original_cfg = FooCfg()
        edited_cfg = transform_configclass_instance(
            original_cfg,
            partial(CompositeTaskBase._remove_configclass_transform, exclude_fields={"field_b"}),
        )

        # Check that remaining fields exist
        assert hasattr(edited_cfg, "field_a")
        assert hasattr(edited_cfg, "field_c")

        # Check that values are preserved
        assert edited_cfg.field_a == 123
        assert edited_cfg.field_c == 1.23

        # Check that removed field doesn't exist
        assert not hasattr(edited_cfg, "field_b")

        # Test removing multiple fields
        original_cfg = FooCfg()
        edited_cfg = transform_configclass_instance(
            original_cfg,
            partial(CompositeTaskBase._remove_configclass_transform, exclude_fields={"field_a", "field_c"}),
        )

        # Check that only field_b remains
        assert hasattr(edited_cfg, "field_b")
        assert edited_cfg.field_b == "123"
        assert not hasattr(edited_cfg, "field_a")
        assert not hasattr(edited_cfg, "field_c")

        # Test None input
        edited_cfg = transform_configclass_instance(
            None,
            partial(CompositeTaskBase._remove_configclass_transform, exclude_fields=set()),
        )
        assert edited_cfg is None

        # Test removing all fields returns None
        original_cfg = FooCfg()
        edited_cfg = transform_configclass_instance(
            original_cfg,
            partial(CompositeTaskBase._remove_configclass_transform, exclude_fields={"field_a", "field_b", "field_c"}),
        )
        assert edited_cfg is None

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


class _MockSuccessFunc:
    """Callable that returns a controlled per-env boolean tensor."""

    def __init__(self, num_envs: int):
        import torch

        self.num_envs = num_envs
        self.return_value = torch.tensor([False] * num_envs)

    def set(self, values: list[bool]):
        import torch

        assert len(values) == self.num_envs
        self.return_value = torch.tensor(values)

    def __call__(self, env, **kwargs):
        return self.return_value


class _MockSubtask:
    """Minimal stand-in for a TaskBase with a controllable success function."""

    def __init__(self, num_envs: int):
        self.func = _MockSuccessFunc(num_envs)

        class _SuccessCfg:
            pass

        class _TerminationCfg:
            pass

        self._termination_cfg = _TerminationCfg()
        self._termination_cfg.success = _SuccessCfg()
        self._termination_cfg.success.func = self.func
        self._termination_cfg.success.params = {}

    def get_termination_cfg(self):
        return self._termination_cfg

    def set_success(self, values: list[bool]):
        self.func.set(values)


class _MockEnv:
    """Minimal stand-in for the env used by composite_task_success_func."""

    def __init__(self, num_envs: int = 1, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.extras = {}


def _test_composite_desired_subtask_success_state_with_none(simulation_app) -> bool:
    """When ``desired_subtask_success_state`` contains None entries, those positions are
    ignored and only positions with True/False are checked. Verifies the composite-task
    matching logic (ordering does not matter for composite tasks)."""

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

    try:
        env = _MockEnv(num_envs=1)
        subtasks = [_MockSubtask(num_envs=1) for _ in range(3)]

        # Latch all three subtasks True simultaneously (composite doesn't require order).
        subtasks[0].set_success([True])
        subtasks[1].set_success([True])
        subtasks[2].set_success([True])
        result = CompositeTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        assert env._subtask_ever_succeeded == [[True, True, True]]
        assert result.tolist() == [True]

        # Subtask 0 currently False (don't-care) -> still success.
        subtasks[0].set_success([False])
        result = CompositeTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        assert result.tolist() == [True]

        # Subtask 2 currently False breaks the [None, True, True] pattern -> failure.
        subtasks[2].set_success([False])
        result = CompositeTaskBase.composite_task_success_func(env, subtasks, [None, True, True])
        assert result.tolist() == [False]

        # [None, False, None]: subtask 1 must be currently False AND latched True at
        # some point. Drive subtask 1 False; it was latched True earlier -> success.
        subtasks[1].set_success([False])
        result = CompositeTaskBase.composite_task_success_func(env, subtasks, [None, False, None])
        assert result.tolist() == [True]

        # All-None desired state matches trivially.
        result = CompositeTaskBase.composite_task_success_func(env, subtasks, [None, None, None])
        assert result.tolist() == [True]

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_composite_desired_subtask_success_state_with_none():
    result = run_simulation_app_function(
        _test_composite_desired_subtask_success_state_with_none,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_composite_desired_subtask_success_state_with_none.__name__} failed"


def test_add_suffix_configclass_transform():
    result = run_simulation_app_function(
        _test_add_suffix_configclass_transform,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_add_suffix_configclass_transform.__name__} failed"


def test_remove_configclass_transform():
    result = run_simulation_app_function(
        _test_remove_configclass_transform,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_remove_configclass_transform.__name__} failed"


if __name__ == "__main__":
    test_add_suffix_configclass_transform()
    test_remove_configclass_transform()
    test_composite_desired_subtask_success_state_with_none()
