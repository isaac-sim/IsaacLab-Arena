# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime drive and authoritative success predicate for gear mesh."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils

from isaaclab_arena.tasks.predicates.gripper import gripper_released

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.gripper import Gripper


def reset_gear_mesh_state(env, env_ids=None) -> None:
    """Clear gear_mesh_success state for standalone or composite tasks."""
    # TODO(cvolk): Temporary CAP workaround for the legacy gear-mesh predicate.
    # Separate motor control/history from success and migrate its hold counter
    # to TrueForConsecutiveStepsCfg, then remove this reset callback.
    progress_tracker = env.progress_tracker
    found_matching_objective = False
    for objective in progress_tracker.progress_objectives:
        # CompositeTaskBase prefixes objective names with the subtask index.
        if objective.name.rsplit("/", 1)[-1] == "gear_mesh":
            gear_mesh_predicate = progress_tracker.get_predicate(objective.name)
            gear_mesh_predicate.reset(env_ids)
            found_matching_objective = True
    # A renamed objective must not silently carry state into the next episode.
    assert found_matching_objective, "reset_gear_mesh_state found no gear_mesh objective to reset."


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


class gear_mesh_success(ManagerTermBase):
    """Latch the motor and require seated rotation after jaw release and withdrawal."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        self.board = env.scene[cfg.params["board_asset_cfg"].name]
        self.gears = tuple(env.scene[asset_cfg.name] for asset_cfg in cfg.params["gear_asset_cfgs"])
        self.gear = self.gears[0]
        assert cfg.params.get("gripper") is not None, "Gear mesh requires a bound embodiment gripper."
        self.pinion_joint = self.board.data.joint_names.index("pinion_joint")
        self.button_joint = self.board.data.joint_names.index("button_joint")
        self.latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.seated_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.started_after_seating = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.success_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.required_steps = math.ceil(float(cfg.params["hold_time_s"]) / float(env.step_dt))
        self.spin_window_steps = max(
            1,
            math.ceil(float(cfg.params.get("spin_window_s", 0.5)) / float(env.step_dt)),
        )
        self.spin_history = torch.zeros((self.spin_window_steps, env.num_envs, len(self.gears)), device=env.device)
        self.spin_samples_seen = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.spin_history_index = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.latched[env_ids] = False
        self.seated_seen[env_ids] = False
        self.started_after_seating[env_ids] = False
        self.success_steps[env_ids] = 0
        self.spin_history[:, env_ids] = 0.0
        self.spin_samples_seen[env_ids] = 0
        if isinstance(env_ids, slice):
            self.spin_history_index = 0

    def __call__(
        self,
        env,
        board_asset_cfg: SceneEntityCfg,
        gear_asset_cfgs: Sequence[SceneEntityCfg],
        gripper: Gripper,
        grasp_width_m: float,
        release_clearance_m: float,
        target_offsets_xyz: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]],
        button_latch_m: float = 0.005,
        drive_speed_rad_s: float = 4.0,
        spin_fraction: float = 0.3,
        gear_teeth: Sequence[int] | Sequence[Sequence[int]] = (20,),
        spin_window_s: float = 0.5,
        xy_threshold_m: float = 0.005348669,
        z_threshold_m: float = 0.008,
        upright_threshold_deg: float = 12.0,
        release_distance_m: float = 0.055,
        hold_time_s: float = 1.0,
    ) -> torch.Tensor:
        del (
            board_asset_cfg,
            gear_asset_cfgs,
            hold_time_s,
            spin_window_s,
        )
        joint_pos = _torch(self.board.data.joint_pos)
        pressed = joint_pos[:, self.button_joint] <= -button_latch_m
        newly_latched = pressed & ~self.latched
        self.latched |= pressed
        target = torch.where(
            self.latched[:, None],
            torch.full_like(joint_pos[:, :1], drive_speed_rad_s),
            torch.zeros_like(joint_pos[:, :1]),
        )
        self.board.set_joint_velocity_target(target, joint_ids=[self.pinion_joint])

        board_pos = _torch(self.board.data.root_pos_w)
        board_quat = _torch(self.board.data.root_quat_w)
        offsets = torch.as_tensor(target_offsets_xyz, device=env.device, dtype=board_pos.dtype)
        if offsets.shape == (len(self.gears), 3):
            offsets = offsets[None].expand(env.num_envs, -1, -1)
        if offsets.shape != (env.num_envs, len(self.gears), 3):
            raise ValueError("gear mesh requires one 3D target offset per station")
        target_pos = board_pos[:, None, :] + math_utils.quat_apply(
            board_quat[:, None, :].expand(-1, len(self.gears), -1).reshape(-1, 4),
            offsets.reshape(-1, 3),
        ).reshape(env.num_envs, len(self.gears), 3)
        gear_pos = torch.stack([_torch(asset.data.root_pos_w) for asset in self.gears], dim=1)
        error = gear_pos[:, None, :, :] - target_pos[:, :, None, :]
        seated = (torch.linalg.vector_norm(error[..., :2], dim=-1) <= xy_threshold_m) & (
            torch.abs(error[..., 2]) <= z_threshold_m
        )

        up = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=board_pos.dtype).expand_as(board_pos)
        gear_quat = torch.stack([_torch(asset.data.root_quat_w) for asset in self.gears], dim=1)
        gear_up = math_utils.quat_apply(
            gear_quat.reshape(-1, 4),
            up[:, None, :].expand(-1, len(self.gears), -1).reshape(-1, 3),
        ).reshape(env.num_envs, len(self.gears), 3)
        board_up = math_utils.quat_apply(board_quat, up)
        seated &= torch.sum(gear_up[:, None, :, :] * board_up[:, None, None, :], dim=-1) >= math.cos(
            math.radians(upright_threshold_deg)
        )
        angular = torch.stack([_torch(asset.data.root_com_vel_w)[:, 3:] for asset in self.gears], dim=1)
        instantaneous_spin = torch.sum(angular * board_up[:, None, :], dim=-1)
        self.spin_history[self.spin_history_index] = instantaneous_spin
        self.spin_history_index = (self.spin_history_index + 1) % self.spin_window_steps
        self.spin_samples_seen = torch.clamp(self.spin_samples_seen + 1, max=self.spin_window_steps)
        windowed_spin = self.spin_history.sum(dim=0) / self.spin_samples_seen[:, None]
        # AUTOLab's board state defines turning as abs(rate) >= gate; the
        # signed rate is diagnostic and does not participate in the goal.
        station_teeth = torch.as_tensor(gear_teeth, device=env.device)
        if station_teeth.shape == (len(self.gears),):
            station_teeth = station_teeth[None].expand(env.num_envs, -1)
        if station_teeth.shape != (env.num_envs, len(self.gears)):
            raise ValueError("gear mesh station count does not match loose gear count")
        chosen = torch.full((env.num_envs, len(self.gears)), -1, dtype=torch.long, device=env.device)
        used = torch.zeros((env.num_envs, len(self.gears)), dtype=torch.bool, device=env.device)
        for station in range(len(self.gears)):
            for gear_index in range(len(self.gears)):
                eligible = (
                    (station_teeth[:, gear_index] == station_teeth[:, station])
                    & seated[:, station, gear_index]
                    & ~used[:, gear_index]
                    & (chosen[:, station] < 0)
                )
                chosen[eligible, station] = gear_index
                used[eligible, gear_index] = True
        assigned = chosen >= 0
        safe_chosen = torch.clamp(chosen, min=0)
        selected_spin = torch.gather(windowed_spin, 1, safe_chosen)
        gates = spin_fraction * drive_speed_rad_s * 14.0 / station_teeth
        turning = assigned & (torch.abs(selected_spin) >= gates)
        # Both gates are intentional: withdrawal alone can pass while the jaws
        # still hold a seated gear, so success also requires physical clearance.
        gripper_clears_gears = gripper_released(
            env,
            gripper=gripper,
            grasp_width_m=grasp_width_m,
            release_clearance_m=release_clearance_m,
        )
        assert (
            math.isfinite(release_distance_m) and release_distance_m >= 0.0
        ), "Release distance must be non-negative and finite."
        gripper_position_w = gripper.get_position_w(env.arena_world)
        gripper_away_by_gear = (
            torch.linalg.vector_norm(gear_pos - gripper_position_w[:, None, :], dim=-1) > release_distance_m
        )
        selected_gripper_away = torch.gather(gripper_away_by_gear, 1, safe_chosen)
        all_seated = assigned.all(dim=1)
        all_valid = gripper_clears_gears & (assigned & turning & selected_gripper_away).all(dim=1)
        self.seated_seen |= all_seated
        self.started_after_seating |= newly_latched & self.seated_seen
        candidate = self.started_after_seating & all_valid
        self.success_steps = torch.where(candidate, self.success_steps + 1, torch.zeros_like(self.success_steps))
        return self.success_steps >= self.required_steps
