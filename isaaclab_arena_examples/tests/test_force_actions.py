# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _force_asset():
    import torch

    def buffer(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))

    asset = SimpleNamespace(
        permanent_wrench_composer=SimpleNamespace(reset=Mock(), set_forces_and_torques_index=Mock()),
        num_joints=1,
        actuators=SimpleNamespace(target_command=SimpleNamespace(set_effort_index=Mock())),
        data=SimpleNamespace(
            root_pose_w=buffer([[0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1]]),
            root_lin_vel_w=buffer([[0, 0, 0], [0, 0, 0]]),
            root_ang_vel_w=buffer([[0, 0, 0], [0, 0, 0]]),
            body_mass=buffer([[1], [2]]),
            body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9).repeat(2, 1, 1) * 0.1),
            joint_pos=buffer([[0], [0]]),
            joint_vel=buffer([[0], [0]]),
        ),
    )
    env = SimpleNamespace(
        num_envs=2, device="cpu", scene={"object": asset}, cfg=SimpleNamespace(sim=SimpleNamespace(gravity=(0, 0, -2)))
    )
    return asset, env


def test_force_pose_recomputes_damping_each_substep_and_releases_selected_envs():
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import (
        ForcePoseActionCfg,
        ForcePoseActionTerm,
    )

    asset, env = _force_asset()
    term = ForcePoseActionTerm(ForcePoseActionCfg(asset_name="object", linear_gain=10, linear_damping=2), env)
    actions = torch.tensor([[1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 0, 1, 1]], dtype=torch.float32)
    term.process_actions(actions)
    term.apply_actions()
    assert term.last_wrench["force"].tolist() == [[10, 0, 2], [20, 0, 4]]
    # No process_actions call: substep feedback must read the new velocity.
    asset.data.root_lin_vel_w.torch[:, 0] = 1
    term.apply_actions()
    assert term.last_wrench["force"].tolist() == [[8, 0, 2], [16, 0, 4]]
    actions[0, 7] = 0
    term.process_actions(actions)
    assert asset.permanent_wrench_composer.reset.call_args.kwargs["env_ids"].tolist() == [0]
    term.apply_actions()
    assert asset.permanent_wrench_composer.set_forces_and_torques_index.call_args.kwargs["env_ids"].tolist() == [1]
    term.reset(env_ids=[1])
    assert not term.processed_actions[:, 7].any()
    assert not term.processed_actions[1].any()
    assert torch.equal(term.processed_actions[0], actions[0])
    assert term.last_wrench is None


@pytest.mark.parametrize(
    ("critical_damping", "expected_force_x"),
    [
        (False, [8.0, 6.0]),
        (True, [3.6754, 1.0557]),
    ],
)
def test_force_pose_selects_linear_damping_mode_independently_of_minimum_stiffness(critical_damping, expected_force_x):
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import (
        ForcePoseActionCfg,
        ForcePoseActionTerm,
    )

    asset, env = _force_asset()
    asset.data.root_lin_vel_w.torch[:, 0] = 1
    cfg = ForcePoseActionCfg(
        asset_name="object",
        linear_gain=0,
        linear_damping=2,
        critical_damping=critical_damping,
        minimum_stiffness=10,
    )
    term = ForcePoseActionTerm(cfg, env)
    term.process_actions(torch.tensor([[1, 0, 1, 0, 0, 0, 1, 1]] * 2, dtype=torch.float32))

    term.apply_actions()

    assert term.last_wrench["force"][:, 0].tolist() == pytest.approx(expected_force_x, abs=1e-4)


@pytest.mark.parametrize("rotation_mode,expected", [("acceleration", 0.1), ("torque", 1.0)])
def test_force_pose_preserves_rotation_modes(rotation_mode, expected):
    import math
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import (
        ForcePoseActionCfg,
        ForcePoseActionTerm,
    )

    _, env = _force_asset()
    term = ForcePoseActionTerm(
        ForcePoseActionCfg(asset_name="object", rotation_mode=rotation_mode, rotation_gain=1), env
    )
    actions = torch.tensor([[0, 0, 1, 0, 0, math.sin(0.5), math.cos(0.5), 1]] * 2)
    term.process_actions(actions)
    term.apply_actions()
    assert term.last_wrench["torque"][:, 2].tolist() == pytest.approx([expected, expected])


def test_joint_efforts_use_substep_feedback_and_release():
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import (
        ForceJointActionCfg,
        ForceJointActionTerm,
    )

    asset, env = _force_asset()
    term = ForceJointActionTerm(ForceJointActionCfg(asset_name="object"), env)
    actions = torch.tensor([[1, 10, 2, 6, 1], [1, 10, 2, 6, 0]], dtype=torch.float32)
    term.process_actions(actions)
    term.apply_actions()
    assert asset.actuators.target_command.set_effort_index.call_args.kwargs["value"].tolist() == [[6]]
    asset.data.joint_pos.torch[0, 0] = 0.75
    asset.data.joint_vel.torch[0, 0] = 0.5
    term.apply_actions()
    assert asset.actuators.target_command.set_effort_index.call_args.kwargs["value"].tolist() == [[1.5]]
    actions[:, 4] = 0
    term.process_actions(actions)
    assert asset.actuators.target_command.set_effort_index.call_args.kwargs["value"].tolist() == [[0]]


def _above_height(env, height):
    return env.scene["cube"].data.root_pos_w.torch[:, 2] > height


def _count_interval(env, env_ids):
    env.force_test_interval_calls = getattr(env, "force_test_interval_calls", 0) + 1


def _cube_pose(env):
    return env.scene["cube"].data.root_pose_w.torch


def _test_force_action_environment_lifecycle(_simulation_app):
    import gymnasium as gym
    import torch

    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg, SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.task_progress_cfg import TaskProgressCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tasks.terminations import task_success_from_progress
    from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import ForcePoseActionCfg

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=CuboidCfg(
                size=(0.1, 0.1, 0.1), rigid_props=RigidBodyPropertiesCfg(), mass_props=MassPropertiesCfg(mass=1)
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 1)),
        )

    @configclass
    class ActionsCfg:
        cube = ForcePoseActionCfg(asset_name="cube")

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            pose = ObservationTermCfg(func=_cube_pose)

        policy = PolicyCfg()

    cfg = IsaacLabArenaManagerBasedRLEnvCfg(
        scene=SceneCfg(num_envs=1, env_spacing=2),
        actions=ActionsCfg(),
        observations=ObservationsCfg(),
        events={"count": EventTermCfg(func=_count_interval, mode="interval", interval_range_s=(0.02, 0.02))},
        terminations={"success": TerminationTermCfg(func=task_success_from_progress)},
        episode_length_s=20,
        decimation=4,
        sim=SimulationCfg(dt=0.005, device="cpu", physics=PhysxCfg()),
        task_progress=TaskProgressCfg(
            success_criteria=[
                CompletionCriteria(
                    name="lift",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(
                            TerminationTermCfg(func=_above_height, params={"height": 1.15}), required_steps=3
                        )
                    ],
                )
            ]
        ),
    )
    env = gym.wrappers.OrderEnforcing(IsaacLabArenaManagerBasedRLEnv(cfg))
    try:
        base = env.unwrapped
        base.configure_episode_limit(1)
        env.reset()
        action = torch.tensor([[0, 0, 1.3, 0, 0, 0, 1, 1]], dtype=torch.float32)
        for _ in range(100):
            env.step(action)
        assert base.common_step_counter == 100
        assert base.episode_length_buf.tolist() == [100]
        assert base.force_test_interval_calls > 0
        assert base.completed_episode_count == 1
        assert base.reset_env_ids.numel() == 0
        assert base.progress_tracker.is_complete().tolist() == [True]
        assert base.scene["cube"].data.root_pos_w.torch[0, 2].item() == pytest.approx(1.3, abs=0.01)
        action[:, 7] = 0
        for _ in range(10):
            env.step(action)
        assert base.scene["cube"].data.root_pos_w.torch[0, 2].item() < 1.2
        assert base.action_manager.get_term("cube").last_wrench is None
    finally:
        env.close()
    return True


def test_force_action_environment_lifecycle():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_force_action_environment_lifecycle)
