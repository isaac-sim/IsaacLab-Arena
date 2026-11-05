# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import math
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class LaptopCupScriptedReplayPolicyCfg(PolicyCfg):
    """Configure a reference demonstration that writes laptop and cup states directly."""

    goal_offset_xyz: tuple[float, float, float] = (-0.02, -0.28, 0.0)
    """Cup root target in the laptop frame, whose positive Y points to the user's left."""

    laptop_open_angle_degrees: float = 105.0
    """Lid angle held during the cup transfer."""

    settling_duration_s: float = 1.0
    """Time for the reset objects to settle before opening the laptop."""

    opening_duration_s: float = 3.0
    """Time spent opening the laptop."""

    open_hold_duration_s: float = 1.0
    """Pause with the laptop open before lifting the cup."""

    cup_lift_height: float = 0.15
    """Cup lift above its settled root height, in meters."""

    front_offset_x: float = -0.24
    """Laptop-frame X coordinate of the cup's path around the front of the laptop."""

    release_clearance: float = 0.005
    """Height above the target at which the cup is released under gravity, in meters."""

    def __post_init__(self) -> None:
        assert len(self.goal_offset_xyz) == 3 and all(math.isfinite(value) for value in self.goal_offset_xyz)
        assert self.goal_offset_xyz[1] < -0.23, "The replay destination must clear the laptop's right edge"
        assert 90.0 <= self.laptop_open_angle_degrees <= 120.0
        assert math.isfinite(self.settling_duration_s) and self.settling_duration_s > 0.0
        assert math.isfinite(self.opening_duration_s) and self.opening_duration_s > 0.0
        assert math.isfinite(self.open_hold_duration_s) and self.open_hold_duration_s >= 0.0
        assert math.isfinite(self.cup_lift_height) and self.cup_lift_height >= 0.10
        assert math.isfinite(self.front_offset_x) and self.front_offset_x <= -0.19
        assert 0.0 < self.release_clearance < self.cup_lift_height


@register_policy
class LaptopCupScriptedReplayPolicy(PolicyBase[LaptopCupScriptedReplayPolicyCfg]):
    """Replay object states with a stationary Franka for a reference demonstration.

    Direct state writes demonstrate the scene sequence; this is not a robot manipulation policy.
    The cup returns to ordinary rigid-body simulation after its release.
    """

    name = "laptop_cup_scripted_replay"

    def __init__(self, config: LaptopCupScriptedReplayPolicyCfg):
        super().__init__(config)
        self._elapsed_s: torch.Tensor | None = None
        self._hinge_joint_ids: list[int] = []

    @torch.inference_mode()
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Advance the object-state replay and hold the Franka end effector in place."""
        arena_env = env.unwrapped
        if self._elapsed_s is None:
            self._initialize(arena_env)

        self._capture_initial_poses(arena_env)
        self._replay_laptop(arena_env)
        self._replay_cup(arena_env)
        self._elapsed_s += arena_env.step_dt

        actions = torch.zeros(env.action_space.shape, device=arena_env.device)
        actions[:, -1] = 1.0
        return actions

    @torch.inference_mode()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart the replay and recapture reset poses for the selected environments."""
        if self._elapsed_s is None:
            return
        selected_env_ids = slice(None) if env_ids is None else env_ids
        self._elapsed_s[selected_env_ids] = 0.0
        self._capture_pending[selected_env_ids] = True
        self._cup_released[selected_env_ids] = False

    def _initialize(self, arena_env) -> None:
        """Check the expected scene and relative Franka controls before replaying states."""
        laptop = arena_env.scene["laptop"]
        arena_env.scene["cup"]
        arena_env.scene["table"]
        robot = arena_env.scene["robot"]
        assert "panda_joint1" in robot.joint_names, "The reference replay requires a Franka robot"
        arm_action = arena_env.cfg.actions.arm_action
        gripper_action = arena_env.cfg.actions.gripper_action
        assert isinstance(arm_action, DifferentialInverseKinematicsActionCfg)
        assert arm_action.controller.command_type == "pose" and arm_action.controller.use_relative_mode
        assert isinstance(gripper_action, BinaryJointPositionActionCfg)
        assert arena_env.action_manager.total_action_dim == 7, "The reference replay requires franka_ik actions"
        self._hinge_joint_ids, _ = laptop.find_joints("lid_hinge")
        assert len(self._hinge_joint_ids) == 1, "The reference replay requires the laptop lid_hinge joint"
        assert len(laptop.joint_names) == 1, "The reference replay expects a single-hinge laptop"

        environment_count = arena_env.num_envs
        device = arena_env.device
        self._elapsed_s = torch.zeros(environment_count, device=device)
        self._capture_pending = torch.ones(environment_count, dtype=torch.bool, device=device)
        self._cup_released = torch.zeros(environment_count, dtype=torch.bool, device=device)
        self._T_W_L = torch.zeros((environment_count, 7), device=device)
        self._T_W_C = torch.zeros((environment_count, 7), device=device)
        self._initial_lid_angles = torch.zeros(environment_count, device=device)
        print("Scripted object-state replay: laptop and cup motion is imposed directly; Franka stays stationary.")

    def _capture_initial_poses(self, arena_env) -> None:
        # W is the simulation world, L the laptop root, and C the cup root.
        laptop = arena_env.scene["laptop"]
        cup = arena_env.scene["cup"]
        capture_env_ids = self._capture_pending.nonzero(as_tuple=False).flatten()
        if capture_env_ids.numel() > 0:
            self._T_W_L[capture_env_ids] = laptop.data.root_pose_w[capture_env_ids]
            self._initial_lid_angles[capture_env_ids] = laptop.data.joint_pos[capture_env_ids, self._hinge_joint_ids[0]]
            self._capture_pending[capture_env_ids] = False

        # Update the starting cup pose through the settling period so the lift begins on the table.
        settling_env_ids = (self._elapsed_s < self.config.settling_duration_s).nonzero(as_tuple=False).flatten()
        if settling_env_ids.numel() > 0:
            self._T_W_C[settling_env_ids] = cup.data.root_pose_w[settling_env_ids]
            local_up = torch.zeros((settling_env_ids.numel(), 3), device=arena_env.device)
            local_up[:, 2] = 1.0
            laptop_up = quat_apply(self._T_W_L[settling_env_ids, 3:], local_up)
            cup_up = quat_apply(self._T_W_C[settling_env_ids, 3:], local_up)
            assert torch.all(laptop_up[:, 2] > 0.99), "The reference replay requires a level laptop"
            assert torch.all(cup_up[:, 2] > 0.98), "The reference replay requires an upright cup"

    def _replay_laptop(self, arena_env) -> None:
        opening_progress = self._smooth_progress(
            (self._elapsed_s - self.config.settling_duration_s) / self.config.opening_duration_s
        )
        open_angle = math.radians(self.config.laptop_open_angle_degrees)
        lid_angles = self._initial_lid_angles + opening_progress * (open_angle - self._initial_lid_angles)
        laptop = arena_env.scene["laptop"]
        laptop.write_joint_state_to_sim(
            lid_angles.unsqueeze(-1),
            torch.zeros_like(lid_angles).unsqueeze(-1),
            joint_ids=self._hinge_joint_ids,
        )

    def _replay_cup(self, arena_env) -> None:
        transfer_start_s = (
            self.config.settling_duration_s + self.config.opening_duration_s + self.config.open_hold_duration_s
        )
        transfer_time_s = self._elapsed_s - transfer_start_s
        active_env_ids = ((transfer_time_s >= 0.0) & ~self._cup_released).nonzero(as_tuple=False).flatten()
        if active_env_ids.numel() == 0:
            return

        T_W_L = self._T_W_L[active_env_ids]
        T_W_C = self._T_W_C[active_env_ids]
        t_L_C_start = quat_apply_inverse(T_W_L[:, 3:], T_W_C[:, :3] - T_W_L[:, :3])
        t_L_C_goal = torch.tensor(self.config.goal_offset_xyz, device=arena_env.device).expand_as(t_L_C_start).clone()
        t_L_C_goal[:, 2] += self.config.release_clearance

        lifted_position = t_L_C_start.clone()
        lifted_position[:, 2] += self.config.cup_lift_height
        front_left_position = lifted_position.clone()
        front_left_position[:, 0] = self.config.front_offset_x
        front_right_position = front_left_position.clone()
        front_right_position[:, 1] = t_L_C_goal[:, 1]
        above_goal_position = t_L_C_goal.clone()
        above_goal_position[:, 2] = lifted_position[:, 2]
        waypoints = (
            t_L_C_start,
            lifted_position,
            front_left_position,
            front_right_position,
            above_goal_position,
            t_L_C_goal,
        )
        segment_durations_s = (1.0, 1.0, 2.0, 1.0, 1.0)
        cup_position = t_L_C_start.clone()
        segment_start_s = 0.0
        for segment_index, segment_duration_s in enumerate(segment_durations_s):
            progress = self._smooth_progress(
                (transfer_time_s[active_env_ids] - segment_start_s) / segment_duration_s
            ).unsqueeze(-1)
            cup_position += progress * (waypoints[segment_index + 1] - waypoints[segment_index])
            segment_start_s += segment_duration_s

        cup_state = torch.zeros((active_env_ids.numel(), 13), device=arena_env.device)
        cup_state[:, :3] = T_W_L[:, :3] + quat_apply(T_W_L[:, 3:], cup_position)
        cup_state[:, 3:7] = T_W_C[:, 3:]
        arena_env.scene["cup"].write_root_state_to_sim(cup_state, env_ids=active_env_ids)

        # The final write releases at rest. Subsequent steps leave cup motion and support to physics.
        released_env_ids = active_env_ids[transfer_time_s[active_env_ids] >= sum(segment_durations_s)]
        self._cup_released[released_env_ids] = True

    @staticmethod
    def _smooth_progress(progress: torch.Tensor) -> torch.Tensor:
        bounded_progress = progress.clamp(0.0, 1.0)
        return bounded_progress.square() * (3.0 - 2.0 * bounded_progress)
