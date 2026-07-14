# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _clear_unit_test_droid_profile():
    from isaaclab_arena.assets.registries import CameraProfileRegistry

    registry = CameraProfileRegistry()
    registry._components.pop("unit_test_droid_profile", None)
    yield
    registry._components.pop("unit_test_droid_profile", None)


def _register_unit_test_droid_profile(calls=None):
    from isaaclab_arena.assets.register import register_camera_profile
    from isaaclab_arena.assets.registries import CameraProfileRegistry
    from isaaclab_arena.embodiments.camera_profile import CameraProfileBase

    calls = [] if calls is None else calls
    registry = CameraProfileRegistry()
    if registry.is_registered("unit_test_droid_profile", ensure_loaded=False):
        return

    @register_camera_profile
    class UnitTestDroidProfile(CameraProfileBase):
        name = "unit_test_droid_profile"
        description = "Unit-test RGB-D profile."
        compatible_embodiments = frozenset({"droid_abs_joint_pos"})

        @classmethod
        def apply(cls, embodiment):
            calls.append(embodiment.name)


def test_camera_profile_registry_applies_compatible_profile():
    from isaaclab_arena.assets.registries import CameraProfileRegistry

    calls = []
    _register_unit_test_droid_profile(calls)
    registry = CameraProfileRegistry()
    assert "unit_test_droid_profile" in registry.get_compatible_profile_names("droid_abs_joint_pos")
    registry.apply_camera_profile("unit_test_droid_profile", "droid_abs_joint_pos", SimpleNamespace(name="robot"))
    assert calls == ["robot"]


def test_camera_profile_registry_rejects_incompatible_profile():
    from isaaclab_arena.assets.registries import CameraProfileRegistry

    _register_unit_test_droid_profile()
    with pytest.raises(AssertionError, match="is not compatible with embodiment"):
        CameraProfileRegistry().apply_camera_profile(
            "unit_test_droid_profile",
            "franka_ik",
            SimpleNamespace(name="robot"),
        )


def test_droid_workspace_exterior_rgbd_profile_is_registered():
    from isaaclab_arena.assets.registries import CameraProfileRegistry

    profile = CameraProfileRegistry().get_camera_profile_by_name("droid_workspace_exterior_rgbd")
    assert profile.name == "droid_workspace_exterior_rgbd"
    assert profile.compatible_embodiments == frozenset({"droid_abs_joint_pos"})
    assert "exterior" in profile.description.lower()
    assert "rgb-d" in profile.description.lower()


def test_maple_gap_profile_uses_named_camera_profile(monkeypatch):
    from isaaclab_arena_environments import pick_and_place_maple_table_environment as maple

    calls = []

    class Registry:
        def apply_camera_profile(self, profile_name, embodiment_registry_name, embodiment):
            calls.append((profile_name, embodiment_registry_name, embodiment.name))

    monkeypatch.setattr(maple, "CameraProfileRegistry", Registry, raising=False)

    embodiment = SimpleNamespace(name="robot")
    maple._apply_gap_camera_profile("droid_abs_joint_pos", embodiment)

    assert calls == [("droid_workspace_exterior_rgbd", "droid_abs_joint_pos", "robot")]
