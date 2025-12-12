# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_add_suffix_to_configclass_fields(simulation_app) -> bool:
    """Test that _add_suffix_to_configclass_fields correctly renames fields with suffix."""

    from isaaclab.utils import configclass

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    @configclass
    class FooCfg:
        int_field: int = 123
        str_field: str = "123"
        float_field: float = 1.23
        bool_field: bool = True

    try:
        original = FooCfg()
        result = SequentialTaskBase._add_suffix_to_configclass_fields(original, "_suffix")

        # Check that new fields exist with suffix
        assert hasattr(result, "int_field_suffix")
        assert hasattr(result, "str_field_suffix")
        assert hasattr(result, "float_field_suffix")
        assert hasattr(result, "bool_field_suffix")

        # Check that values are preserved
        assert result.int_field_suffix == 123
        assert result.str_field_suffix == "123"
        assert result.float_field_suffix == 1.23
        assert result.bool_field_suffix is True

        # Check types are preserved
        assert isinstance(result.int_field_suffix, int)
        assert isinstance(result.str_field_suffix, str)
        assert isinstance(result.float_field_suffix, float)
        assert isinstance(result.bool_field_suffix, bool)

        # Check that old field names don't exist
        assert not hasattr(result, "int_field")
        assert not hasattr(result, "str_field")
        assert not hasattr(result, "float_field")
        assert not hasattr(result, "bool_field")

        # Test None input
        result_none = SequentialTaskBase._add_suffix_to_configclass_fields(None, "_suffix")
        assert result_none is None

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def _test_remove_configclass_fields(simulation_app) -> bool:
    """Test that _remove_configclass_fields correctly removes specified fields."""

    from isaaclab.utils import configclass

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    @configclass
    class OriginalCfg:
        field_a: int = 123
        field_b: str = "123"
        field_c: float = 1.23

    try:
        original = OriginalCfg()
        result = SequentialTaskBase._remove_configclass_fields(original, {"field_b"})

        # Check that remaining fields exist
        assert hasattr(result, "field_a")
        assert hasattr(result, "field_c")

        # Check that values are preserved
        assert result.field_a == 123
        assert result.field_c == 1.23

        # Check that removed field doesn't exist
        assert not hasattr(result, "field_b")

        # Test removing multiple fields
        original2 = OriginalCfg()
        result2 = SequentialTaskBase._remove_configclass_fields(original2, {"field_a", "field_c"})

        # Check that only field_b remains
        assert hasattr(result2, "field_b")
        assert result2.field_b == "123"
        assert not hasattr(result2, "field_a")
        assert not hasattr(result2, "field_c")

        # Test None input
        result_none = SequentialTaskBase._remove_configclass_fields(None, set())
        assert result_none is None

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def test_add_suffix_to_configclass_fields():
    result = run_simulation_app_function(
        _test_add_suffix_to_configclass_fields,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_add_suffix_to_configclass_fields.__name__} failed"


def test_remove_configclass_fields():
    result = run_simulation_app_function(
        _test_remove_configclass_fields,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_remove_configclass_fields.__name__} failed"


if __name__ == "__main__":
    test_add_suffix_to_configclass_fields()
    test_remove_configclass_fields()
