# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the ported USB-C task semantics and reset ranges."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _test_usbc_insertion_task(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.registries import EnvironmentRegistry, TaskRegistry
    from isaaclab_arena.tasks.predicates.spatial import (
        depth_in_range,
        lateral_in_proximity,
        tilt_axis_aligned,
        velocity_below_threshold,
    )
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tasks.terminations import check_success
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.task import UsbcInsertionTask

    class _World:
        def __init__(self):
            self.poses = {
                "plug": torch.tensor([[0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                "receiver": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            }
            self.velocity = torch.zeros((1, 3))

        def get_pose_w(self, name):
            return self.poses[name]

        def get_root_linear_velocity_w(self, _name):
            return self.velocity

    env = SimpleNamespace(num_envs=1, device="cpu", arena_world=_World())
    mating = {
        "subject_name": "plug",
        "receiver_name": "receiver",
        "subject_offset_xyz": (0.0, 0.0, 0.0),
        "target_offset_xyz": (0.0, 0.0, 0.0),
        "receiver_axis": (1.0, 0.0, 0.0),
    }
    assert depth_in_range(env, **mating, depth_min=0.01, depth_max=0.02).item()
    assert lateral_in_proximity(env, **mating, tolerance_lateral=0.001).item()
    assert tilt_axis_aligned(
        env,
        subject_name="plug",
        receiver_name="receiver",
        subject_axis=(1.0, 0.0, 0.0),
        receiver_axis=(1.0, 0.0, 0.0),
        max_tilt_rad=0.01,
    ).item()
    env.arena_world.poses["plug"][0, 1] = 0.002
    assert not lateral_in_proximity(env, **mating, tolerance_lateral=0.001).item()

    easy_task = UsbcInsertionTask(
        Asset("plug"),
        Asset("receiver"),
        receiver_mouth_offset_xyz=(-0.0045, 0.0, 0.0),
        receiver_axis=(1.0, 0.0, 0.0),
        subject_tip_offset_xyz=(0.0, 0.0, 0.0133),
        depth_min=0.0104,
        lateral_max=0.0087931792,
        speed_max=0.05,
        tilt_max=0.0873,
        allow_antiparallel_axes=True,
        episode_length_s=150.0,
    )
    termination_cfg = easy_task.get_termination_cfg()
    assert termination_cfg.timeout_s == 150.0
    assert len(termination_cfg.success) == 1
    assert termination_cfg.success[0].name == "usbc_insertion"
    success_requirement = termination_cfg.success[0].predicate_sequence[0]
    assert isinstance(success_requirement, TrueForConsecutiveStepsCfg)
    assert success_requirement.required_steps == 1
    assert success_requirement.predicate.func is check_success
    assert success_requirement.predicate.to_dict()["func"] == "isaaclab_arena.tasks.terminations:check_success"
    predicates = success_requirement.predicate.params["predicates"]
    assert [predicate.func for predicate in predicates] == [
        depth_in_range,
        lateral_in_proximity,
        tilt_axis_aligned,
        velocity_below_threshold,
    ]
    assert predicates[0].params["depth_max"] is None

    medium_task = UsbcInsertionTask(
        Asset("plug"),
        Asset("receiver"),
        receiver_mouth_offset_xyz=(0.0, 0.0, 0.0065),
        receiver_axis=(0.0, 0.0, -1.0),
        subject_tip_offset_xyz=(0.0, 0.0, 0.00665),
        depth_min=0.0052,
        depth_max=0.0085,
        lateral_max=0.0043965896,
        speed_max=0.05,
        consecutive_success_steps=3,
    )
    medium_requirement = medium_task.get_termination_cfg().success[0].predicate_sequence[0]
    assert medium_requirement.required_steps == 3
    predicates = medium_requirement.predicate.params["predicates"]
    assert [predicate.func for predicate in predicates] == [
        depth_in_range,
        lateral_in_proximity,
        velocity_below_threshold,
    ]
    assert predicates[0].params["depth_max"] == 0.0085

    register_components()
    assert TaskRegistry().get_task_by_name("UsbcInsertionTask") is UsbcInsertionTask
    environment_registry = EnvironmentRegistry()
    for name in (
        "vabar_contact_rich_insertion__usbc_insertion_easy",
        "vabar_contact_rich_insertion__usbc_insertion_medium",
    ):
        assert environment_registry.is_registered(name)
    return True


def test_usbc_insertion_task() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_insertion_task)


def _test_usbc_release_and_withdrawal(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.tasks.predicates.gripper import gripper_released
    from isaaclab_arena.tasks.predicates.spatial import gripper_distance_from_object_exceeds_threshold
    from isaaclab_arena_environments.isaac_cap.embodiments.cable_routing.gripper import YamGripper
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.task import UsbcInsertionTask

    gripper = YamGripper()
    positions = torch.tensor([[0.0, 0.0, 0.13], [0.0, 0.0, 0.33], [0.0, 0.0, 0.33]])
    frame_positions = torch.tensor([[0.0, 0.0, 0.13]]).expand(3, -1)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_names=["left_finger"],
            joint_pos=torch.tensor([[0.037524], [0.005], [0.037524]]),
        )
    )
    env = SimpleNamespace(
        arena_world=SimpleNamespace(
            get_joint_position=lambda _robot, _joint: robot.data.joint_pos[:, 0],
            get_frame_position_w=lambda _sensor, _target: frame_positions,
            get_pose_w=lambda _name: positions,
        ),
    )
    release_params = dict(
        gripper=gripper,
        grasp_width_m=0.01,
        release_clearance_m=0.0015,
    )
    params = dict(
        subject_name="plug",
        gripper=gripper,
        distance_threshold_m=0.04,
    )
    assert gripper.get_jaw_gap_m(env.arena_world).tolist() == pytest.approx([0.075048, 0.01, 0.075048])
    assert gripper_released(env, **release_params).tolist() == [True, False, True]
    assert gripper_distance_from_object_exceeds_threshold(env, **params).tolist() == [False, True, True]
    assert (
        gripper_released(env, **release_params) & gripper_distance_from_object_exceeds_threshold(env, **params)
    ).tolist() == [
        False,
        False,
        True,
    ]
    for require_released in (False, True):
        task = UsbcInsertionTask(
            Asset("plug"),
            Asset("receiver"),
            receiver_mouth_offset_xyz=(0.0, 0.0, 0.0),
            receiver_axis=(0.0, 0.0, 1.0),
            subject_tip_offset_xyz=(0.0, 0.0, 0.0),
            depth_min=0.01,
            lateral_max=0.01,
            speed_max=0.05,
            grasp_width_m=0.01,
            release_clearance_m=0.0015,
            withdrawal_distance_min=0.04,
            require_released=require_released,
        )
        success_requirement = task.get_termination_cfg().success[0].predicate_sequence[0]
        hand_predicates = success_requirement.predicate.params["predicates"][3:]
        assert [term.func for term in hand_predicates] == (
            [gripper_released, gripper_distance_from_object_exceeds_threshold]
            if require_released
            else [gripper_distance_from_object_exceeds_threshold]
        )
        assert all("gripper" not in term.params for term in hand_predicates)
        task.configure_for_embodiment(SimpleNamespace(get_gripper=lambda: gripper))
        assert all(term.params["gripper"] is gripper for term in hand_predicates)
        result = torch.stack([term.func(env, **term.params) for term in hand_predicates]).all(dim=0)
        assert result.tolist() == ([False, False, True] if require_released else [False, True, True])
    robot.data.joint_pos[1, 0] = 0.006
    assert gripper_released(env, **release_params).tolist() == [True, True, True]
    assert (
        gripper_released(env, **release_params) & gripper_distance_from_object_exceeds_threshold(env, **params)
    ).tolist() == [
        False,
        True,
        True,
    ]
    positions[0] = frame_positions[0] + torch.tensor([0.04, 0.0, 0.0])
    assert not gripper_distance_from_object_exceeds_threshold(env, **params)[0]
    positions[0, 0] += 1.0e-4
    assert gripper_distance_from_object_exceeds_threshold(env, **params)[0]
    return True


def test_usbc_release_and_withdrawal() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_release_and_withdrawal)


def _test_usbc_contact_rig(_simulation_app) -> bool:
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import patch

    import newton
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_arena_environments.isaac_cap.usbc_insertion.physics import _configure_contacts

    labels = [
        "/World/envs/env_0/LeftRobot/left_finger",
        "/World/envs/env_0/RightRobot/left_finger",
        "/World/envs/env_0/LeftRobot/link_6",
        "/World/envs/env_0/RightRobot/link_6",
        "/World/envs/env_0/RightRobot/wrist_support/geometry",
        "/World/envs/env_0/Plug/geometry",
        "/World/envs/env_0/Port/geometry",
        "/World/envs/env_0/Bulkhead/geometry",
        "/World/envs/env_0/Bench/geometry",
        "/World/envs/env_0/background/table/geometry",
        "/World/envs/env_0/UsbcConnectorCablePlug/segment0",
    ]
    count = len(labels)
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    builder = SimpleNamespace(
        body_label=labels,
        joint_label=["LeftRobot/left_finger", "RightRobot/left_finger"],
        shape_label=labels,
        shape_body=list(range(count)),
        shape_source=[object()] * count,
        shape_flags=[collide, collide, 0, 0, collide, collide, collide, collide, collide, collide],
        shape_collision_group=[1] * count,
        _shape_collision_filter_pairs=set(),
        custom_attributes={
            name: SimpleNamespace(values=None)
            for name in (
                "mujoco:eq_solref",
                "mujoco:solref",
                "mujoco:solref_mode",
                "mujoco:geom_solimp",
                "mujoco:condim",
            )
        },
    )
    builder.custom_attributes["mujoco:equality_constraint_joint1"] = SimpleNamespace(values=[0, 1])
    builder.add_shape_collision_filter_pair = lambda first, second: builder._shape_collision_filter_pairs.add(
        (first, second)
    )
    for name in ("mu", "mu_torsional", "mu_rolling", "ke", "kd"):
        setattr(builder, f"shape_material_{name}", [0.0] * count)
    builder.shape_gap = [0.0] * count
    with patch.object(NewtonManager, "_builder", builder):
        _configure_contacts()
    np.testing.assert_allclose(builder.shape_material_mu, [8, 8, 0, 0, 0, 0.35, 0.35, 2.5, 0.4, 0.35, 0])
    np.testing.assert_allclose(builder.shape_material_mu_torsional[:2], [0.002, 0.002])
    np.testing.assert_allclose(builder.shape_gap[:2], [0.0002, 0.0002])
    np.testing.assert_allclose(builder.shape_material_ke[5:8], [62500] * 3)
    np.testing.assert_allclose(builder.shape_material_kd[5:8], [500] * 3)
    assert all(builder.shape_flags[index] & collide for index in (2, 3))
    assert builder.custom_attributes["mujoco:condim"].values == {0: 4, 1: 4, 2: 3, 3: 3}
    for value in builder.custom_attributes["mujoco:eq_solref"].values.values():
        np.testing.assert_allclose(tuple(value), [0.004, 1.0])
    assert 4 not in builder.custom_attributes["mujoco:solref"].values
    assert 10 not in builder.custom_attributes["mujoco:solref"].values
    return True


def test_usbc_contact_rig() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_contact_rig)


def _test_usbc_asset_registration(_simulation_app) -> bool:
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.assets import ASSET_ROOT, USBC_ASSET_CLASSES
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )

    register_components()
    register_components()
    registry = AssetRegistry()
    assert len(USBC_ASSET_CLASSES) == 11
    for asset_class in USBC_ASSET_CLASSES:
        assert registry.get_asset_by_name(asset_class.name) is asset_class
        assert "usbc_insertion" in asset_class.tags
        if "light" not in asset_class.tags and "cable" not in asset_class.tags:
            assert asset_class.usd_path.startswith(f"{ASSET_ROOT}/")

    for unused_name in ("plug", "precision_plug", "port", "fr3_table"):
        assert not registry.is_registered(f"usbc_insertion_{unused_name}", ensure_loaded=False)

    for name in ("easy_port", "bench", "cradle_front", "cradle_rear"):
        fixture = registry.get_asset_by_name(f"usbc_insertion_{name}")()
        assert fixture.object_type == ObjectType.RIGID
        assert fixture.object_cfg.spawn.rigid_props.kinematic_enabled
        assert fixture.reset_pose

    easy = UsbcInsertionEasyEnvironment().build(UsbcInsertionEasyEnvironmentCfg())
    medium = UsbcInsertionMediumEnvironment().build(UsbcInsertionMediumEnvironmentCfg())
    asset_types = {type(asset) for env in (easy, medium) for asset in env.scene.assets.values()}
    assert set(USBC_ASSET_CLASSES).issuperset(asset_types - {type(easy.scene.assets["table"])})
    assert type(easy.task.plug) is registry.get_asset_by_name("usbc_insertion_easy_plug")
    assert type(medium.task.plug) is registry.get_asset_by_name("usbc_insertion_medium_plug")
    assert easy.task.plug.name == "plug" and easy.task.receiver.name == "port"
    assert medium.task.plug.name == "plug" and medium.task.receiver.name == "bulkhead"
    for scene in (easy.scene, medium.scene):
        assert scene.assets["table"].object_type == ObjectType.BASE
        assert scene.assets["background"].prim_path == "{ENV_REGEX_NS}/background"
        shadow_receiver = scene.assets["hdr_shadow_receiver"]
        assert not shadow_receiver.object_cfg.spawn.visible
        assert shadow_receiver.object_cfg.collision_group == -1
    return True


def test_usbc_asset_registration() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_asset_registration)


def _test_usbc_environment_yaml(_simulation_app) -> bool:
    import math

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.predicates.gripper import gripper_released
    from isaaclab_arena.tasks.predicates.spatial import (
        depth_in_range,
        gripper_distance_from_object_exceeds_threshold,
        lateral_in_proximity,
        velocity_below_threshold,
    )
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena_environments.isaac_cap.embodiments.cable_routing import IndustrialBimanualYamEmbodiment
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.assets import ASSET_ROOT
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.physics import NewtonUsbcManager

    yam = IndustrialBimanualYamEmbodiment(
        robot_usd_path=f"{ASSET_ROOT}/industrial__i2rt_yam/i2rt_yam_default.usda",
        instanceable_robot_usd_path=f"{ASSET_ROOT}/industrial__i2rt_yam/i2rt_yam_instanceable.usda",
        left_mount_position=(0.2525, 0.31, 0.75),
        right_mount_position=(0.2525, -0.31, 0.75),
    )
    assert yam.get_ee_frame_transformer_names() == []
    assert yam.scene_config.left_ee_frame is yam.scene_config.right_ee_frame is None
    assert yam.get_ee_frame_name(ArmMode.DUAL_ARM) == "link_6"
    assert yam.get_gripper() is yam.gripper
    assert yam.gripper.articulation_name == "right_robot"
    assert yam.gripper.driver_joint_name == "left_finger"
    assert yam.gripper.frame_transformer_name == "right_ee_frame"
    assert yam.gripper.target_frame_name == "tcp"

    for factory, cfg_type, variant in (
        (UsbcInsertionEasyEnvironment(), UsbcInsertionEasyEnvironmentCfg, "easy"),
        (UsbcInsertionMediumEnvironment(), UsbcInsertionMediumEnvironmentCfg, "medium"),
    ):
        environment = factory.build(cfg_type(enable_cameras=True, use_tiled_cameras=True, use_instanceable_meshes=True))
        spec = ArenaEnvGraphSpec.from_yaml(factory.scene_spec)
        assert environment.name == factory.name
        assert environment.task.task_description == "insert the USB-C plug into the receptacle"
        assert spec.embodiment.registry_name == "industrial_bimanual_yam"
        assert ASSET_ROOT.startswith("https://omniverse-content-staging.s3-us-west-2.amazonaws.com/")
        assert spec.embodiment.params["robot_usd_path"] == f"{ASSET_ROOT}/industrial__i2rt_yam/i2rt_yam_default.usda"
        assert (
            spec.embodiment.params["instanceable_robot_usd_path"]
            == f"{ASSET_ROOT}/industrial__i2rt_yam/i2rt_yam_instanceable.usda"
        )
        assert spec.default_physics_backend is PhysicsBackend.NEWTON
        assert spec.env_cfg_override is not None
        assert not environment.placer_params.allow_best_loss_fallbacks
        assert environment.placer_params.required_checks == {"on_relation"}
        assert environment.placer_params.solver_params.clearance_m == 0.0
        assert environment.placer_params.solver_params.lr == 0.001
        assert environment.task.plug is environment.scene.assets["plug"]
        assert environment.task.receiver is environment.scene.assets["port" if variant == "easy" else "bulkhead"]
        assert environment.scene.assets["background"].usd_path.startswith(f"{ASSET_ROOT}/")
        assert environment.task.plug.usd_path.startswith(f"{ASSET_ROOT}/")
        assert environment.task.receiver.usd_path.startswith(f"{ASSET_ROOT}/")
        assert environment.task.plug.scale == (1.0, 1.0, 1.0)
        assert environment.task.get_events_cfg() is None
        success_objective = environment.task.get_termination_cfg().success[0]
        assert success_objective.name == "usbc_insertion"
        predicates = success_objective.predicate_sequence[0].predicate.params["predicates"]
        assert [term.func for term in predicates] == [
            depth_in_range,
            lateral_in_proximity,
            velocity_below_threshold,
            gripper_released,
            gripper_distance_from_object_exceeds_threshold,
        ]
        assert predicates[0].params["depth_min"] == 0.0104
        assert predicates[0].params["depth_max"] is None
        assert "gripper" not in predicates[-2].params
        assert "gripper" not in predicates[-1].params
        environment.task.configure_for_embodiment(environment.embodiment)
        assert predicates[-2].params == {
            "grasp_width_m": 0.01,
            "release_clearance_m": 0.0015,
            "gripper": environment.embodiment.gripper,
        }
        assert predicates[-1].params["subject_name"] == "plug"
        assert predicates[-1].params["gripper"] is environment.embodiment.gripper
        assert predicates[-1].params["distance_threshold_m"] == (0.04 if variant == "easy" else 0.05)
        randomization = {
            relation.subject: relation.params
            for relation in spec.relations
            if relation.kind == "random_around_solution"
        }
        assert randomization["plug"]["x_half_m"] == (0.012 if variant == "easy" else 0.018)
        assert randomization["plug"]["y_half_m"] == randomization["plug"]["x_half_m"]
        assert randomization["plug"]["yaw_half_rad"] == pytest.approx(math.radians(8 if variant == "easy" else 50))
        events = environment.scene.get_events_cfg()
        cable = environment.scene.assets["plug_cable"]
        assert getattr(events, "plug_cable").params["links"] == 16
        assert cable.get_event_cfg()[1].mode == "reset"
        if variant == "easy":
            assert not {"bulkhead", "cradle_front", "cradle_rear", "bulkhead_cable"} & environment.scene.assets.keys()
            assert environment.task.receiver.get_initial_pose().position_xyz == (0.44, 0.0233, 0.8234)
        else:
            assert events.bulkhead_cable.params["links"] == 8
            assert randomization["bulkhead"]["x_half_m"] == 0.004
            assert randomization["bulkhead"]["yaw_half_rad"] == pytest.approx(math.radians(3))
            supports = [
                relation for relation in spec.relations if relation.subject == "bulkhead" and relation.kind == "on"
            ]
            assert len(supports) == 1 and supports[0].reference == "cradle_front"
            assert supports[0].params["overlap"] is True
        cameras = environment.embodiment.camera_config
        assert cameras.use_tiled_camera
        assert cameras.top_camera.offset.pos == (0.44, 0.0, 1.2)
        assert cameras.right_wrist_camera.offset.pos == (-0.0017, 0.079729, 0.066021)
        assert cameras.right_wrist_camera.update_latest_camera_pose
        assert cameras.right_wrist_camera.width == 320
        assert environment.embodiment.get_ee_frame_transformer_names() == ["left_ee_frame", "right_ee_frame"]
        for side in ("Left", "Right"):
            frame = getattr(environment.embodiment.scene_config, f"{side.lower()}_ee_frame")
            assert frame.prim_path == f"{{ENV_REGEX_NS}}/{side}Robot/Geometry/arm"
            assert len(frame.target_frames) == 1
            target = frame.target_frames[0]
            assert (
                target.prim_path
                == f"{{ENV_REGEX_NS}}/{side}Robot/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"
            )
            assert target.name == "tcp"
            assert target.offset.pos == (0.0, -0.044, 0.13)
        env_cfg, _ = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=1)).compose_manager_cfg()
        assert env_cfg.sim.physics.class_type is NewtonUsbcManager
        assert env_cfg.sim.dt == 1.0 / 60.0 and env_cfg.decimation == 1
        assert env_cfg.sim.physics.num_substeps == 16
        assert env_cfg.sim.physics.collision_decimation == 1
        assert env_cfg.sim.physics.solver_cfg.integrator == "implicitfast"
        assert env_cfg.sim.physics.solver_cfg.impratio == 10.0
        assert not env_cfg.sim.physics.solver_cfg.use_mujoco_contacts
        assert not env_cfg.sim.physics.use_cuda_graph
        for robot in (env_cfg.scene.left_robot, env_cfg.scene.right_robot):
            assert robot.actuators["arm_joints_1_3"].stiffness == 1600.0
            assert robot.actuators["gripper"].stiffness == 40000.0
            assert robot.actuators["gripper"].damping == 40.0
            assert robot.actuators["gripper"].effort_limit_sim == 160.0
            assert robot.init_state.joint_pos["joint2"] == 1.047
            assert robot.spawn.usd_path.startswith(f"{ASSET_ROOT}/")
    return True


def test_usbc_environment_yaml() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_environment_yaml)


def _test_usbc_cable_hook_cleanup(_simulation_app) -> bool:
    from functools import partial
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_newton.physics import NewtonManager

    from isaaclab_arena_environments.isaac_cap.usbc_insertion.cables import (
        _add_connector_cable,
        _remove_connector_cable_builder_hooks,
    )

    def unrelated_hook(*_args):
        return None

    hooks = [
        unrelated_hook,
        partial(_add_connector_cable, cfg=SimpleNamespace(attachment="plug")),
        partial(_add_connector_cable, cfg=SimpleNamespace(attachment="bulkhead")),
    ]
    with patch.object(NewtonManager, "_per_world_builder_hooks", hooks):
        _remove_connector_cable_builder_hooks()
        assert NewtonManager._per_world_builder_hooks == [unrelated_hook]
    return True


def test_usbc_cable_hook_cleanup() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_cable_hook_cleanup)


def _test_usbc_cable_reset_isolation(_simulation_app) -> bool:
    import numpy as np
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_arena_environments.isaac_cap.usbc_insertion.cables import reset_connector_cable

    labels = [
        f"/World/envs/env_{world}/{name}/bend{joint}"
        for world in range(2)
        for name in ("TestCable", "TestCableOther")
        for joint in range(2)
    ]
    model = SimpleNamespace(
        joint_label=labels,
        joint_q_start=wp.array(np.arange(9, dtype=np.int32), device="cpu"),
        joint_qd_start=wp.array(np.arange(9, dtype=np.int32), device="cpu"),
    )
    states = [
        SimpleNamespace(
            joint_q=wp.array(np.full(8, 0.25, dtype=np.float32), device="cpu"),
            joint_qd=wp.array(np.full(8, 0.1, dtype=np.float32), device="cpu"),
        )
        for _ in range(2)
    ]
    with (
        patch.object(NewtonManager, "get_model", return_value=model),
        patch.object(NewtonManager, "get_state_0", return_value=states[0]),
        patch.object(NewtonManager, "get_state_1", return_value=states[1]),
        patch.object(NewtonManager, "invalidate_fk") as invalidate,
    ):
        env = SimpleNamespace(num_envs=2)
        params = {"prim_path": "{ENV_REGEX_NS}/TestCable", "links": 2}
        reset_connector_cable(env, [], **params)
        invalidate.assert_not_called()
        reset_connector_cable(env, torch.tensor([0]), **params)
        invalidate.assert_called_once()
        for state in states:
            np.testing.assert_allclose(state.joint_q.numpy(), [0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            np.testing.assert_allclose(state.joint_qd.numpy(), [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        reset_connector_cable(env, None, **params)
        for state in states:
            np.testing.assert_allclose(state.joint_q.numpy(), [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25])
            np.testing.assert_allclose(state.joint_qd.numpy(), [0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1])
    return True


def test_usbc_cable_reset_isolation() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_cable_reset_isolation)


def _check_usbc_cable_reset(base_env, arena_environment) -> None:
    import numpy as np
    import torch

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    states = (NewtonManager.get_state_0(), NewtonManager.get_state_1())
    coordinate_starts = model.joint_q_start.numpy()
    velocity_starts = model.joint_qd_start.numpy()
    cable_indices = [index for index, label in enumerate(model.joint_label) if "UsbcConnectorCable" in label]
    snapshots = []
    for state in states:
        coordinates = state.joint_q.numpy()
        velocities = state.joint_qd.numpy()
        for index in cable_indices:
            coordinates[coordinate_starts[index] : coordinate_starts[index + 1]] = 0.25
            velocities[velocity_starts[index] : velocity_starts[index + 1]] = 0.1
        state.joint_q.assign(coordinates)
        state.joint_qd.assign(velocities)
        snapshots.append((coordinates.copy(), velocities.copy()))

    _, event = arena_environment.scene.assets["plug_cable"].get_event_cfg()
    assert "reset_usbc_cables" not in base_env.event_manager.active_terms["reset"]
    event.func(base_env, [], **event.params)
    for state, (coordinates, velocities) in zip(states, snapshots, strict=True):
        np.testing.assert_array_equal(state.joint_q.numpy(), coordinates)
        np.testing.assert_array_equal(state.joint_qd.numpy(), velocities)

    event.func(base_env, torch.tensor([0], device=base_env.device), **event.params)
    for state, (coordinates, velocities) in zip(states, snapshots, strict=True):
        for index in cable_indices:
            if str(model.joint_label[index]).startswith("/World/envs/env_0/UsbcConnectorCablePlug/bend"):
                coordinates[coordinate_starts[index] : coordinate_starts[index + 1]] = 0.0
                velocities[velocity_starts[index] : velocity_starts[index + 1]] = 0.0
        np.testing.assert_array_equal(state.joint_q.numpy(), coordinates)
        np.testing.assert_array_equal(state.joint_qd.numpy(), velocities)

    base_env._reset_idx(torch.tensor([0], device=base_env.device))
    for state in states:
        coordinates = state.joint_q.numpy()
        velocities = state.joint_qd.numpy()
        for index in cable_indices:
            reset = str(model.joint_label[index]).startswith("/World/envs/env_0/")
            np.testing.assert_allclose(
                coordinates[coordinate_starts[index] : coordinate_starts[index + 1]], 0.0 if reset else 0.25
            )
            np.testing.assert_allclose(
                velocities[velocity_starts[index] : velocity_starts[index + 1]], 0.0 if reset else 0.1
            )
    base_env._reset_idx(torch.arange(base_env.num_envs, device=base_env.device))


def _test_usbc_insertion_environment(_simulation_app, variant: str, num_envs: int = 1) -> bool:
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )

    if variant == "easy":
        arena_environment = UsbcInsertionEasyEnvironment().build(UsbcInsertionEasyEnvironmentCfg())
    else:
        assert variant == "medium"
        arena_environment = UsbcInsertionMediumEnvironment().build(UsbcInsertionMediumEnvironmentCfg())

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    env_builder = ArenaEnvBuilder(arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    try:
        env.reset()
        base_env = env.unwrapped
        plug_pose = base_env.arena_world.get_pose_w(arena_environment.task.plug.name)
        receiver_pose = base_env.arena_world.get_pose_w(arena_environment.task.receiver.name)
        assert plug_pose.shape == receiver_pose.shape == (num_envs, 7)
        assert torch.isfinite(plug_pose).all()
        assert torch.isfinite(receiver_pose).all()
        from isaaclab_newton.physics import NewtonManager

        half_range = 0.012 if variant == "easy" else 0.018
        plug_positions = plug_pose[:, :3] - base_env.scene.env_origins
        assert torch.all(torch.abs(plug_positions[:, 0] - 0.44) <= half_range + 0.002)
        assert torch.all(torch.abs(plug_positions[:, 1] + 0.07) <= half_range + 0.002)
        cable_joints = [label for label in NewtonManager.get_model().joint_label if "UsbcConnectorCable" in label]
        assert len(cable_joints) == (16 if variant == "easy" else 24) * num_envs
        _check_usbc_cable_reset(base_env, arena_environment)

        receiver = base_env.scene[arena_environment.task.receiver.name]
        moved_pose = receiver_pose.clone()
        moved_pose[:, 0] += 0.05
        receiver.write_root_pose_to_sim(moved_pose)
        base_env._reset_idx(torch.arange(num_envs, device=base_env.device))
        restored_pose = base_env.arena_world.get_pose_w(arena_environment.task.receiver.name)
        if variant == "easy":
            assert torch.allclose(restored_pose[:, :3], receiver_pose[:, :3], atol=1.0e-6)
        else:
            local_positions = restored_pose[:, :3] - base_env.scene.env_origins
            assert torch.all(torch.abs(local_positions[:, 0] - 0.44) <= 0.006)
            assert torch.all(torch.abs(local_positions[:, 1] - 0.0148) <= 0.006)

        action = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(action)
        success = base_env.termination_manager.get_term("success")
        assert success.shape == (num_envs,)
    finally:
        env.close()
    return True


def test_usbc_insertion_medium_environment() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_usbc_insertion_environment,
        variant="medium",
    )


def test_usbc_insertion_easy_environment() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_usbc_insertion_environment,
        variant="easy",
    )
