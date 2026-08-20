# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_runtime_object_hierarchy(simulation_app):
    """Runtime object classes expose the transform and provenance mixins they support."""
    from isaaclab_arena.assets.deformable_object import DeformableObject, DeformableTransform
    from isaaclab_arena.assets.object import Object, RootedTransform, SpawnPrim
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.assets.object_reference import ObjectReference, ReferencedPrim

    assert issubclass(Object, ObjectBase)
    assert SpawnPrim in Object.__mro__
    assert RootedTransform in Object.__mro__

    assert issubclass(ObjectReference, ObjectBase)
    assert ReferencedPrim in ObjectReference.__mro__
    assert RootedTransform in ObjectReference.__mro__

    assert issubclass(DeformableObject, ObjectBase)
    assert SpawnPrim in DeformableObject.__mro__
    assert DeformableTransform in DeformableObject.__mro__
    assert RootedTransform not in DeformableObject.__mro__
    return True


def test_runtime_object_hierarchy():
    assert run_function_with_persistent_simulation_app(_test_runtime_object_hierarchy, headless=HEADLESS)


def _test_runtime_object_affordance_mro_initialization(simulation_app):
    """Rooted runtime classes cooperatively initialize affordances through their MRO."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.affordances.openable import Openable
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import OpenableObjectReference, ReferencedPrim

    class OpenableObject(Object, Openable):
        def _generate_base_cfg(self):
            return SimpleNamespace(init_state=SimpleNamespace())

    rooted_object = OpenableObject(
        name="cabinet",
        usd_path="/tmp/cabinet.usd",
        object_type=ObjectType.BASE,
        openable_joint_name="door_joint",
        openable_threshold=0.25,
    )
    assert rooted_object.name == "cabinet"
    assert rooted_object.openable_joint_name == "door_joint"
    assert rooted_object.openable_threshold == 0.25

    with patch.object(ReferencedPrim, "_init_referenced_prim", lambda self, parent_asset: None):
        rooted_reference = OpenableObjectReference(
            name="door",
            prim_path="{ENV_REGEX_NS}/cabinet/door",
            parent_asset=rooted_object,
            openable_joint_name="door_joint",
            openable_threshold=0.75,
        )
    assert rooted_reference.name == "door"
    assert rooted_reference.openable_joint_name == "door_joint"
    assert rooted_reference.openable_threshold == 0.75
    return True


def test_runtime_object_affordance_mro_initialization():
    assert run_function_with_persistent_simulation_app(
        _test_runtime_object_affordance_mro_initialization,
        headless=HEADLESS,
    )


def _test_affordance_base(simulation_app):

    from isaaclab_arena.affordances.openable import Openable
    from isaaclab_arena.assets.asset import Asset

    class NotAnAsset:

        def __init__(self, blah: str, **kwargs):
            super().__init__(**kwargs)
            self.blah = blah

    class OpenableAsset(Asset, Openable):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class OpenableNotAnAsset(NotAnAsset, Openable):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    _ = OpenableAsset(name="test_name", openable_joint_name="test_joint_name", openable_threshold=0.5)

    with pytest.raises(TypeError) as exception_info:
        _ = OpenableNotAnAsset(blah="test_name", openable_joint_name="test_joint_name", openable_threshold=0.5)
    assert "must inherit from Asset" in str(exception_info.value)

    return True


def test_affordance_base():
    result = run_function_with_persistent_simulation_app(
        _test_affordance_base,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_affordance_base()
