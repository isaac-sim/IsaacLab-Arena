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
    _configure_cap_environment_profile,
    _configure_cap_grocery_embodiment,
    _disable_cap_automatic_camera_rendering,
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
from isaaclab_arena.scripts.run_cap_barrier_move_to_pose_serve import (
    _PerceptionStreamSink,
    _close_resources,
    _run_serve,
)


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
    assert embodiment.camera_config.external_camera is None
    assert embodiment.camera_config.external_camera_2 is None
    assert embodiment.camera_config.wrist_camera is None


def test_camera_profile_removes_automatic_observation_group() -> None:
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.02),
        decimation=4,
        observations=SimpleNamespace(camera_obs=object(), policy=object()),
    )

    configured = _configure_cap_environment_profile(cfg, enable_cameras=True)

    assert configured is cfg
    assert cfg.sim.dt == 0.005
    assert cfg.decimation == 1
    assert cfg.observations.camera_obs is None
    assert cfg.observations.policy is not None


def test_camera_profile_fails_if_camera_observation_contract_is_missing() -> None:
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.02),
        decimation=4,
        observations=SimpleNamespace(policy=object()),
    )

    with pytest.raises(RuntimeError, match="camera_obs"):
        _configure_cap_environment_profile(cfg, enable_cameras=True)


def test_camera_profile_disables_kit_rendering_on_control_steps() -> None:
    environment = SimpleNamespace(unwrapped=SimpleNamespace(render_enabled=True))

    _disable_cap_automatic_camera_rendering(environment)

    assert environment.unwrapped.render_enabled is False


def test_camera_profile_rejects_environment_without_render_control() -> None:
    with pytest.raises(RuntimeError, match="render_enabled"):
        _disable_cap_automatic_camera_rendering(SimpleNamespace(unwrapped=SimpleNamespace()))


def test_resource_cleanup_continues_after_perception_close_failure() -> None:
    closed: list[str] = []

    class _Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self._name = name
            self._fail = fail

        def close(self) -> None:
            closed.append(self._name)
            if self._fail:
                raise RuntimeError(f"{self._name} close failed")

    with pytest.raises(RuntimeError, match="perception close failed"):
        _close_resources(
            _Resource("perception", fail=True),
            _Resource("barrier-client"),
            _Resource("adapter"),
        )

    assert closed == ["perception", "barrier-client", "adapter"]


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
    assert kwargs["perception_frames_per_generation"] == 1
    assert kwargs["perception_first_capture_generation"] == 2
    assert kwargs["serve_seconds"] == 321.0
    assert kwargs["initial_gripper_closed"] is False
    assert kwargs["ready_marker"].endswith("camera_profile=oblique")
    assert kwargs["environment_factory"].func is make_cap_grocery_to_bin_environment
    assert kwargs["environment_factory"].keywords == {"camera_profile": "oblique"}
    assert calls[3] == ("context_exit", True)


def test_grocery_perception_skips_pre_reset_and_captures_once_after_reset(monkeypatch) -> None:
    import isaaclab_arena.integrations.cap_barrier.perception_producer as perception_module

    offered: list[int] = []
    extracted: list[int] = []

    class _Producer:
        def __init__(self, *, endpoint: str) -> None:
            assert endpoint == "127.0.0.1:50061"

        def start(self) -> None:
            pass

        def offer(self, frame) -> bool:
            offered.append(frame.frame_index)
            return True

        @property
        def stats(self):
            return {"offered": len(offered), "sent": len(offered), "dropped": 0, "stream_starts": 1}

        def close(self) -> None:
            pass

    def _extract(_environment, *, frame_index: int):
        extracted.append(frame_index)
        return SimpleNamespace(frame_index=frame_index)

    monkeypatch.setattr(perception_module, "PerceptionFrameProducer", _Producer)
    monkeypatch.setattr(perception_module, "extract_camera_frame", _extract)
    markers: list[str] = []
    sink = _PerceptionStreamSink(
        SimpleNamespace(_environment=object()),
        "127.0.0.1:50061",
        markers.append,
        frames_per_generation=1,
        first_capture_generation=2,
    )

    sink.begin_generation(1)
    for frame in range(100):
        sink.on_physics_frame(frame)
    sink.begin_generation(2)
    for frame in range(100):
        sink.on_physics_frame(frame)
    sink.close()

    assert extracted == [0]
    assert offered == [0]
    assert sum("PERCEPTION_GENERATION_CAPTURED" in marker for marker in markers) == 1


def test_uncapped_perception_stream_keeps_decimated_capture_behavior(monkeypatch) -> None:
    import isaaclab_arena.integrations.cap_barrier.perception_producer as perception_module

    extracted: list[int] = []

    class _Producer:
        def __init__(self, *, endpoint: str) -> None:
            pass

        def start(self) -> None:
            pass

        def offer(self, frame) -> bool:
            return True

        def close(self) -> None:
            pass

        @property
        def stats(self):
            return {"offered": len(extracted), "sent": len(extracted), "dropped": 0, "stream_starts": 1}

    def _extract(_environment, *, frame_index: int):
        extracted.append(frame_index)
        return SimpleNamespace(frame_index=frame_index)

    monkeypatch.setattr(perception_module, "PerceptionFrameProducer", _Producer)
    monkeypatch.setattr(perception_module, "extract_camera_frame", _extract)
    sink = _PerceptionStreamSink(
        SimpleNamespace(_environment=object()),
        "127.0.0.1:50061",
        lambda _marker: None,
        frames_per_generation=None,
    )
    sink.begin_generation(1)

    for frame in range(45):
        sink.on_physics_frame(frame)
    sink.close()

    assert extracted == [0, 1, 2]


def test_perception_capture_requires_monotonic_generations(monkeypatch) -> None:
    import isaaclab_arena.integrations.cap_barrier.perception_producer as perception_module

    class _Producer:
        def __init__(self, *, endpoint: str) -> None:
            pass

        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        @property
        def stats(self):
            return {"offered": 0, "sent": 0, "dropped": 0, "stream_starts": 0}

    monkeypatch.setattr(perception_module, "PerceptionFrameProducer", _Producer)
    sink = _PerceptionStreamSink(
        SimpleNamespace(_environment=object()),
        "127.0.0.1:50061",
        lambda _marker: None,
        frames_per_generation=1,
    )
    sink.begin_generation(2)
    with pytest.raises(ValueError, match="must advance"):
        sink.begin_generation(2)
    sink.close()


def test_bounded_perception_capture_does_not_retry_failed_rtx_read(monkeypatch) -> None:
    import isaaclab_arena.integrations.cap_barrier.perception_producer as perception_module

    attempts: list[int] = []

    class _Producer:
        def __init__(self, *, endpoint: str) -> None:
            pass

        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        @property
        def stats(self):
            return {"offered": 0, "sent": 0, "dropped": 0, "stream_starts": 0}

    def _failed_extract(_environment, *, frame_index: int):
        attempts.append(frame_index)
        raise RuntimeError("synthetic RTX failure")

    monkeypatch.setattr(perception_module, "PerceptionFrameProducer", _Producer)
    monkeypatch.setattr(perception_module, "extract_camera_frame", _failed_extract)
    markers: list[str] = []
    sink = _PerceptionStreamSink(
        SimpleNamespace(_environment=object()),
        "127.0.0.1:50061",
        markers.append,
        frames_per_generation=1,
    )
    sink.begin_generation(1)

    for frame in range(100):
        sink.on_physics_frame(frame)
    sink.close()

    assert attempts == [0]
    assert sum("PERCEPTION_SAMPLE_FAILED" in marker for marker in markers) == 1
