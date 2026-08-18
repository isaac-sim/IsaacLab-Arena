# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted pick-and-place oracle for generating demonstrations.

A privileged waypoint controller: it reads object and destination poses straight from the scene and
drives the end effector through hover → descend → close → lift → transport → release. Intended as a
*demonstration generator*, not a policy under test — it is allowed to cheat, because the point is to
produce trajectories a learned policy can be trained on.

Two things make it more than a convenience:

* It aligns the jaws with the object's **narrow** footprint axis, so every demonstration it produces
  is attainable by construction (``a|cos phi| + b|sin phi| <= stroke``). Generating in the infeasible
  band would teach a policy to attempt grasps physics forbids.
* It runs on ``droid_differential_ik``, whose action term does the IK internally, so no solver is
  needed here. Recorded joint trajectories transfer to the ``droid_abs_joint_pos`` embodiment used
  for evaluation: same robot, same cameras, same scene — only the action term differs.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

# Phase order. Each advances when its goal is met, or when it times out.
HOVER, DESCEND, CLOSE, LIFT, TRANSPORT, RELEASE, DONE = range(7)
_PHASE_NAMES = ("hover", "descend", "close", "lift", "transport", "release", "done")


@dataclass
class ScriptedPickPlacePolicyCfg(PolicyCfg):
    """Waypoint geometry and gains. Distances in metres."""

    object_name: str = "sugar_box_ycb_robolab"
    destination_name: str = "bowl_ycb_robolab"
    hover_height: float = 0.18
    """Height above the object to reach before descending."""
    grasp_height: float = 0.02
    """Fingertip height above the object origin at which the jaws close."""
    lift_height: float = 0.22
    transport_height: float = 0.22
    position_gain: float = 4.0
    """Delta-pose command per metre of position error, before the action term's own scale."""
    yaw_gain: float = 2.0
    max_step: float = 0.9
    """Clip on the commanded delta, in action units."""
    position_tolerance: float = 0.015
    yaw_tolerance: float = 0.10
    settle_steps: int = 12
    """Steps to hold while the gripper opens or closes before advancing."""
    phase_timeout: int = 180


@register_policy
class ScriptedPickPlacePolicy(PolicyBase[ScriptedPickPlacePolicyCfg]):
    """Privileged waypoint controller producing attainable pick-and-place demonstrations."""

    name = "scripted_pick_place"

    def __init__(self, config: ScriptedPickPlacePolicyCfg):
        super().__init__(config)
        self._phase: torch.Tensor | None = None
        self._phase_step: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._phase is None:
            return
        if env_ids is None:
            self._phase[:] = HOVER
            self._phase_step[:] = 0
        else:
            self._phase[env_ids] = HOVER
            self._phase_step[env_ids] = 0

    # --- scene readers (privileged by design) --------------------------------------------------

    def _fingertip_mid_and_yaw(self, scene) -> tuple[torch.Tensor, torch.Tensor]:
        """Fingertip midpoint and jaw-axis yaw, both from the frame transformer, indexed by name.

        Index 0 on DROID is ``end_effector``, which resolves to the robot base — using it here would
        silently steer from the wrong body.
        """
        import warp as wp

        frame = scene["ee_frame"]
        names = list(frame.data.target_frame_names)
        li, ri = names.index("tool_leftfinger"), names.index("tool_rightfinger")
        pos = wp.to_torch(frame.data.target_pos_w)
        left, right = pos[:, li, :], pos[:, ri, :]
        d = right - left
        return 0.5 * (left + right), torch.atan2(d[:, 1], d[:, 0])

    @staticmethod
    def _yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
        x, y, z, w = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        import warp as wp

        base = env.unwrapped
        scene, device = base.scene, torch.device(base.device)
        n = base.num_envs
        cfg = self.config

        if self._phase is None:
            self._phase = torch.full((n,), HOVER, dtype=torch.long, device=device)
            self._phase_step = torch.zeros((n,), dtype=torch.long, device=device)

        obj_pose = wp.to_torch(scene[cfg.object_name].data.root_link_pose_w)
        obj_pos, obj_yaw = obj_pose[:, :3], self._yaw_from_quat_xyzw(obj_pose[:, 3:7])
        dest_pos = wp.to_torch(scene[cfg.destination_name].data.root_link_pose_w)[:, :3]
        ee_pos, jaw_yaw = self._fingertip_mid_and_yaw(scene)

        # Per-phase goal position, and whether the jaws should be closed.
        goal = torch.zeros((n, 3), device=device)
        close = torch.zeros((n,), device=device)
        for ph, tgt, grip in (
            (HOVER, obj_pos + torch.tensor([0.0, 0.0, cfg.hover_height], device=device), 0.0),
            (DESCEND, obj_pos + torch.tensor([0.0, 0.0, cfg.grasp_height], device=device), 0.0),
            (CLOSE, obj_pos + torch.tensor([0.0, 0.0, cfg.grasp_height], device=device), 1.0),
            (LIFT, obj_pos + torch.tensor([0.0, 0.0, cfg.lift_height], device=device), 1.0),
            (TRANSPORT, dest_pos + torch.tensor([0.0, 0.0, cfg.transport_height], device=device), 1.0),
            (RELEASE, dest_pos + torch.tensor([0.0, 0.0, cfg.transport_height], device=device), 0.0),
            (DONE, dest_pos + torch.tensor([0.0, 0.0, cfg.transport_height], device=device), 0.0),
        ):
            m = self._phase == ph
            goal[m], close[m] = tgt[m], grip

        # Align the jaws with the object's narrow footprint axis: that is the only orientation whose
        # closable span fits inside the gripper stroke.
        yaw_err = torch.atan2(torch.sin(obj_yaw - jaw_yaw), torch.cos(obj_yaw - jaw_yaw))

        pos_err = goal - ee_pos
        action = torch.zeros((n, 7), device=device)
        action[:, :3] = (cfg.position_gain * pos_err).clamp(-cfg.max_step, cfg.max_step)
        # Only correct yaw while still above the object; rotating mid-grasp would twist it free.
        pre_grasp = (self._phase == HOVER) | (self._phase == DESCEND)
        action[:, 5] = torch.where(pre_grasp, (cfg.yaw_gain * yaw_err).clamp(-cfg.max_step, cfg.max_step), torch.zeros_like(yaw_err))
        action[:, 6] = close

        # Advance phases.
        at_goal = pos_err.norm(dim=-1) < cfg.position_tolerance
        aligned = yaw_err.abs() < cfg.yaw_tolerance
        self._phase_step += 1
        ready = torch.where(
            (self._phase == CLOSE) | (self._phase == RELEASE),
            self._phase_step >= cfg.settle_steps,
            at_goal & (aligned | ~pre_grasp),
        )
        advance = (ready | (self._phase_step >= cfg.phase_timeout)) & (self._phase < DONE)
        self._phase = torch.where(advance, self._phase + 1, self._phase)
        self._phase_step = torch.where(advance, torch.zeros_like(self._phase_step), self._phase_step)
        return action

    def phase_names(self) -> list[str]:
        """Current phase per env, for logging."""
        return [_PHASE_NAMES[i] for i in (self._phase.tolist() if self._phase is not None else [])]
