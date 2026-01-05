# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_add_suffix_configclass_transform(simulation_app) -> bool:
    """Test that add_suffix_configclass_transform correctly renames fields with suffix."""

    from functools import partial

    from isaaclab.utils import configclass

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
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
            partial(SequentialTaskBase.add_suffix_configclass_transform, suffix="_suffix"),
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
            partial(SequentialTaskBase.add_suffix_configclass_transform, suffix="_suffix"),
        )
        assert edited_cfg is None

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def _test_remove_configclass_transform(simulation_app) -> bool:
    """Test that remove_configclass_transform correctly removes specified fields."""

    from functools import partial

    from isaaclab.utils import configclass

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
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
            partial(SequentialTaskBase.remove_configclass_transform, exclude_fields={"field_b"}),
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
            partial(SequentialTaskBase.remove_configclass_transform, exclude_fields={"field_a", "field_c"}),
        )

        # Check that only field_b remains
        assert hasattr(edited_cfg, "field_b")
        assert edited_cfg.field_b == "123"
        assert not hasattr(edited_cfg, "field_a")
        assert not hasattr(edited_cfg, "field_c")

        # Test None input
        edited_cfg = transform_configclass_instance(
            None,
            partial(SequentialTaskBase.remove_configclass_transform, exclude_fields=set()),
        )
        assert edited_cfg is None

        # Test removing all fields returns None
        original_cfg = FooCfg()
        edited_cfg = transform_configclass_instance(
            original_cfg,
            partial(SequentialTaskBase.remove_configclass_transform, exclude_fields={"field_a", "field_b", "field_c"}),
        )
        assert edited_cfg is None

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


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
