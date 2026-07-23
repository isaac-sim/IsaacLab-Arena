# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import inspect
import math
from types import SimpleNamespace

import isaaclab.sim as sim_utils
import pytest

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.integrations.cap_barrier.franka_env import (
    _configure_cap_camera,
    _configure_cap_grocery_embodiment,
    _make_cap_grocery_assets,
    make_cap_grocery_to_bin_environment,
)
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_BIN_POSE,
    CAP_GROCERY_CAMERA_NAME,
    CAP_GROCERY_DROID_HOME,
    CAP_GROCERY_OBJECT_ASSET,
    CAP_GROCERY_OBJECT_POSE,
    CAP_GROCERY_SUPPORT_INSTANCE,
    CAP_GROCERY_SUPPORT_POSE,
    CAP_GROCERY_SUPPORT_SIZE,
)
from isaaclab_arena.scripts.run_cap_barrier_grocery_to_bin import (
    _add_grocery_arguments,
    _environment_factory,
    _run_grocery,
    _scene_ready_marker,
)
from isaaclab_arena.scripts.run_cap_barrier_move_to_pose_serve import _run_serve


def _droid_embodiment():
    return AssetRegistry().get_asset_by_name("droid_abs_joint_pos")(enable_cameras=False)


def test_grocery_embodiment_preserves_identity_base_and_pins_proven_home() -> None:
    embodiment = _droid_embodiment()

    _configure_cap_grocery_embodiment(embodiment)

    assert tuple(embodiment.scene_config.robot.init_state.pos) == (0.0, 0.0, 0.0)
    assert tuple(embodiment.scene_config.robot.init_state.rot) == (0.0, 0.0, 0.0, 1.0)
    assert tuple(embodiment.event_config.init_franka_arm_pose.params["default_pose"]) == CAP_GROCERY_DROID_HOME


def test_grocery_embodiment_rejects_a_base_pose_outside_pinned_calibration() -> None:
    embodiment = _droid_embodiment()
    embodiment.scene_config.robot.init_state.pos = (0.2, 0.0, 0.0)

    with pytest.raises(RuntimeError, match="identity DROID base"):
        _configure_cap_grocery_embodiment(embodiment)


def test_grocery_scene_uses_dynamic_proven_assets_and_fixed_reset_poses() -> None:
    assets = _make_cap_grocery_assets(AssetRegistry(), sim_utils)
    scene_assets = {asset.name: asset for asset in assets}

    assert set(scene_assets) == {
        "ground_plane",
        "light",
        CAP_GROCERY_SUPPORT_INSTANCE,
        CAP_GROCERY_BIN_ASSET,
        CAP_GROCERY_OBJECT_ASSET,
    }
    grocery = scene_assets[CAP_GROCERY_OBJECT_ASSET]
    destination = scene_assets[CAP_GROCERY_BIN_ASSET]
    support = scene_assets[CAP_GROCERY_SUPPORT_INSTANCE]

    assert grocery.object_type == ObjectType.RIGID
    assert {"object", "graspable", "food"}.issubset(grocery.tags)
    assert grocery.object_cfg.spawn.rigid_props is None
    assert grocery.get_initial_pose().position_xyz == CAP_GROCERY_OBJECT_POSE[0]
    assert grocery.get_initial_pose().rotation_xyzw == CAP_GROCERY_OBJECT_POSE[1]
    assert grocery.get_event_cfg()[1] is not None

    assert destination.object_type == ObjectType.RIGID
    assert {"object", "container"}.issubset(destination.tags)
    assert destination.object_cfg.spawn.rigid_props is None
    assert destination.get_initial_pose().position_xyz == CAP_GROCERY_BIN_POSE[0]
    assert destination.get_initial_pose().rotation_xyzw == CAP_GROCERY_BIN_POSE[1]
    assert destination.get_event_cfg()[1] is not None

    assert support.get_initial_pose().position_xyz == CAP_GROCERY_SUPPORT_POSE[0]
    assert tuple(support.object_cfg.spawn.size) == CAP_GROCERY_SUPPORT_SIZE
    assert support.object_cfg.spawn.visible is False
    assert support.object_cfg.spawn.rigid_props.kinematic_enabled is True


def test_proven_layout_is_inside_support_and_recorded_radial_envelope() -> None:
    support_x, support_y, _ = CAP_GROCERY_SUPPORT_POSE[0]
    half_x = CAP_GROCERY_SUPPORT_SIZE[0] * 0.5
    half_y = CAP_GROCERY_SUPPORT_SIZE[1] * 0.5

    for position in (CAP_GROCERY_OBJECT_POSE[0], CAP_GROCERY_BIN_POSE[0]):
        assert support_x - half_x < position[0] < support_x + half_x
        assert support_y - half_y < position[1] < support_y + half_y
        assert math.hypot(position[0], position[1]) < 0.5


@pytest.mark.parametrize(
    ("profile", "class_name"),
    [
        ("libero", "LiberoDroidPerceptionCameraCfg"),
        ("oblique", "MapleDroidPerceptionCameraCfg"),
    ],
)
def test_camera_profiles_are_explicit_and_publish_exterior_camera(profile: str, class_name: str) -> None:
    embodiment = _droid_embodiment()

    _configure_cap_camera(embodiment, profile)

    assert type(embodiment.camera_config).__name__ == class_name
    assert embodiment.camera_config.exterior_cam.prim_path.endswith(f"/{CAP_GROCERY_CAMERA_NAME}")
    assert embodiment.camera_config.exterior_cam.update_latest_camera_pose is True
    assert set(embodiment.camera_config.exterior_cam.data_types) == {
        "rgb",
        "distance_to_image_plane",
    }


def test_unknown_camera_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported CAP grocery camera profile"):
        _configure_cap_camera(_droid_embodiment(), "silent-fallback")


def test_grocery_runner_requires_perception_and_has_bounded_camera_choices() -> None:
    parser = argparse.ArgumentParser()
    _add_grocery_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--perception-stream", "127.0.0.1:50061", "--camera", "oblique"])

    assert args.perception_stream == "127.0.0.1:50061"
    assert args.camera == "oblique"
    with pytest.raises(SystemExit):
        parser.parse_args(["--perception-stream", "127.0.0.1:50061", "--camera", "unknown"])


def test_grocery_runner_injects_only_scene_specific_serve_configuration() -> None:
    factory = _environment_factory("libero")
    assert factory.func is make_cap_grocery_to_bin_environment
    assert factory.keywords == {"camera_profile": "libero"}
    assert (
        _scene_ready_marker("libero")
        == "CAP_GROCERY_TO_BIN_SCENE_READY "
        "object=alphabet_soup_can_hope_robolab "
        "bin=grey_bin_robolab "
        "camera=exterior_cam camera_profile=libero"
    )

    defaults = inspect.signature(_run_serve).parameters
    assert defaults["environment_factory"].default is None
    assert defaults["initial_gripper_closed"].default is True
    assert defaults["ready_marker"].default == "CAP_SERVE_KIT_ARM_READY_FOR_MOVE_TO_POSE"


def test_grocery_runner_executes_required_scene_and_perception_wiring() -> None:
    calls: list[tuple[str, object]] = []

    class _Context:
        def __init__(self, args) -> None:
            calls.append(("context_init", args.enable_cameras))

        def __enter__(self):
            calls.append(("context_enter", True))

        def __exit__(self, *_args) -> None:
            calls.append(("context_exit", True))

    def serve(device, **kwargs) -> None:
        calls.append(("serve", (device, kwargs)))

    args = SimpleNamespace(
        device="cuda:0",
        perception_stream="127.0.0.1:50061",
        serve_seconds=321.0,
        camera="oblique",
        enable_cameras=False,
    )

    _run_grocery(args, context_factory=_Context, serve=serve)

    assert args.enable_cameras is True
    assert calls[:2] == [("context_init", True), ("context_enter", True)]
    _, (device, kwargs) = calls[2]
    assert device == "cuda:0"
    assert kwargs["perception_stream"] == "127.0.0.1:50061"
    assert kwargs["serve_seconds"] == 321.0
    assert kwargs["initial_gripper_closed"] is False
    assert kwargs["ready_marker"].endswith("camera_profile=oblique")
    assert kwargs["environment_factory"].func is make_cap_grocery_to_bin_environment
    assert kwargs["environment_factory"].keywords == {"camera_profile": "oblique"}
    assert calls[3] == ("context_exit", True)
