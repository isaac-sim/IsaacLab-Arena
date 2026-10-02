# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify CAP syringe scene definitions and FR3 socket configuration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap
_SYRINGE_ROOT = Path(__file__).parents[1] / "syringe_sort"


def _test_syringe_cap_experiments(_simulation_app):
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.embodiments.droid.droid import BinaryJointPositionZeroToOneActionCfg
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicyCfg
    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments.cameras import SyringeCameraCfg

    for variant, scene_count, scored_count in (
        ("single", 1, 1),
        ("both", 2, 2),
        ("designated", 2, 1),
        ("cluttered", 6, 6),
    ):
        experiment = load_arena_experiment_from_config_file(
            _SYRINGE_ROOT / "experiment_configs" / f"{variant}_cap_remote_experiment.yaml", device="cuda:0"
        )
        run = experiment.runs[variant]
        assert isinstance(run.policy, CapPolicyCfg)
        assert run.policy.workspace == {
            "surface_z": -0.117,
            "transport_z": 0.19,
            "align_clearance_m": 0.12,
            "pregrasp_standoff_m": 0.12,
        }
        assert run.policy.gripper_closed_position == 0.8
        assert run.policy.startup_render_steps == 0
        registry = EnvironmentRegistry()
        factory = registry.get_component_by_name(f"syringe_{variant}_newton")
        environment = factory().build(registry.get_environment_cfg_type(factory)())
        assert isinstance(environment.embodiment.camera_config, SyringeCameraCfg)
        assert not environment.embodiment.camera_config.use_tiled_camera
        syringes = {name: asset for name, asset in environment.scene.assets.items() if name.startswith("syringe_")}
        assert len(syringes) == scene_count
        assert [obj.name for obj in environment.task.objects] == [f"syringe_{i}" for i in range(scored_count)]
        for name, asset in syringes.items():
            expected = "syringe_blank" if variant == "designated" and name == "syringe_1" else "syringe"
            assert f"vabar_tool_sort__{expected}/" in asset.usd_path
        assert environment.task.episode_length_s == 228 * scored_count
        assert environment.task.consecutive_success_steps == 50
        assert environment.task.gripper_open_position_threshold == 0.1
        assert environment.placer_params.max_placement_attempts == 10
        assert environment.placer_params.solver_params.clearance_m == 0.01
        assert not environment.placer_params.allow_best_loss_fallbacks
        gripper = environment.embodiment.scene_config.robot.actuators["robotiq_driver"]
        assert (gripper.stiffness, gripper.damping) == (100, 10)
        assert isinstance(environment.embodiment.action_config.gripper_action, BinaryJointPositionZeroToOneActionCfg)
    return True


def test_syringe_cap_experiments():
    assert run_function_with_persistent_simulation_app(_test_syringe_cap_experiments)


def _test_syringe_camera_config_isolation(_simulation_app):
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    from isaaclab_arena_environments.isaac_cap.embodiments.insertion_task.cameras import IndustrialFr3RobotiqCameraCfg
    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments.cameras import SyringeCameraCfg

    shared = IndustrialFr3RobotiqCameraCfg()
    shared_before = shared.to_dict()
    cameras = SyringeCameraCfg()
    calibrated = cameras.to_dict()
    assert set(cameras.camera_names()) == {
        "top_camera",
        "wrist_camera",
        "exterior_left_camera",
        "exterior_right_camera",
    }
    for tiled in (False, True):
        cameras.use_tiled_camera = tiled
        before_resolving = cameras.to_dict()
        resolved = cameras.get_cfg()
        for name in cameras.camera_names():
            camera = getattr(resolved, name)
            assert isinstance(camera, CameraCfg)
            assert isinstance(camera, TiledCameraCfg) == tiled
            # Scene assembly and camera variations may mutate resolved configs.
            camera.offset.pos = (9.0, 9.0, 9.0)
            camera.spawn.focal_length += 1.0
            camera.data_types.append("normals")
        assert cameras.to_dict() == before_resolving

    # Changing one environment's calibration must not affect another or the FR3 defaults.
    cameras.wrist_camera.offset.pos = (9.0, 9.0, 9.0)
    assert SyringeCameraCfg().to_dict() == calibrated
    assert shared.to_dict() == shared_before
    assert IndustrialFr3RobotiqCameraCfg().to_dict() == shared_before
    return True


def test_syringe_camera_config_isolation():
    assert run_function_with_persistent_simulation_app(_test_syringe_camera_config_isolation)


def _test_syringe_socket_contract(_simulation_app):
    import numpy as np
    import torch

    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    workspace = {"surface_z": -0.117, "transport_z": 0.19, "align_clearance_m": 0.12, "pregrasp_standoff_m": 0.12}
    policy = CapPolicy(CapPolicyCfg(gripper_closed_position=0.8, workspace=workspace, startup_render_steps=0))
    # A binary gripper has no action scale. Observation normalization comes from the experiment.
    joints = torch.tensor([[0.4, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
    robot = SimpleNamespace(
        joint_names=["left_driver_joint", *(f"fr3_joint{i}" for i in range(1, 8))],
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joints)),
    )
    env = SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=1, step_dt=0.02, cap_episode_finished=False)
    policy._camera = lambda _env, name: {"camera": name}
    policy._tip_reach = lambda _robot: 0.02
    action = policy._hold_action(env)
    assert action.tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 7, 0.5])
    frame = policy._observation_frame(env, action)
    assert frame["left"]["joint_pos"] == pytest.approx([1, 2, 3, 4, 5, 6, 7, 0.5])
    assert frame["_isaac_cap"]["workspace"] == workspace
    assert frame["overhead"] == {"camera": "top_camera"}
    assert frame["eye_in_hand"] == {"camera": "wrist_camera"}
    assert frame["agentview"] == {"camera": "exterior_left_camera"}
    # The first exchange must use the image produced by environment reset.
    # Extra rendering changes SAM detections even without moving an object.
    env.unwrapped = env
    env.sim = SimpleNamespace(render=lambda: pytest.fail("Syringe reset already refreshed camera images"))
    exchanged = []
    policy._connect = lambda: None
    policy._exchange = lambda packet: exchanged.append(packet) or {}
    policy.get_action(env, {})
    assert len(exchanged) == 1
    for name in ("overhead", "eye_in_hand", "agentview", "left", "_isaac_cap"):
        assert exchanged[0][name] == frame[name]
    policy._apply_reply({"left": {"gripper": 1.0}, "arm_valid": False, "gripper_valid": True}, action)
    assert action.tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 7, 0])
    policy._apply_reply({"left": {"joint_pos": np.arange(7)}, "arm_valid": True, "gripper_valid": False}, action)
    assert action.tolist() == pytest.approx([0, 1, 2, 3, 4, 5, 6, 0])
    # Completion permits a settling window and ends the episode without asserting success.
    for _ in range(100):
        policy._settle(env, action)
    assert env.cap_episode_finished
    policy._env = env
    policy.reset()
    assert not env.cap_episode_finished
    return True


def test_syringe_socket_contract():
    assert run_function_with_persistent_simulation_app(_test_syringe_socket_contract)


def _test_syringe_tray_matches_cap_collision(_simulation_app):
    from isaaclab.sim.spawners.from_files import UsdFileCfg
    from isaaclab.sim.utils import create_new_stage
    from pxr import UsdPhysics

    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments.assets import InstrumentTray, SharpsContainer

    create_new_stage()
    cfg = UsdFileCfg(usd_path=InstrumentTray.usd_path, **InstrumentTray.spawn_cfg_addon)
    prim = cfg.func("/World/SyringeTray", cfg)
    body = prim.GetChild("Geometry").GetChild("instrument_tray_01_obj_00")
    assert UsdPhysics.CollisionAPI(body.GetChild("instrument_tray_01_mesh_00")).GetCollisionEnabledAttr().Get()
    assert not body.GetChild("CavityCollision").IsActive()
    assert "func" not in SharpsContainer.spawn_cfg_addon
    return True


def test_syringe_tray_matches_cap_collision():
    assert run_function_with_persistent_simulation_app(_test_syringe_tray_matches_cap_collision)
