# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted privileged controllers for the spatial-memory suite.

* ``SpatialMemoryOraclePolicy`` reads the ground-truth layout, including hidden geometry, and follows the task's
  oracle waypoint plan. It shows that a policy with a perfect map can solve the task.
* ``SpatialMemoryBlindPolicy`` uses the same skills but ignores the hidden geometry (straight-line motions or a
  nominal-layout guess). It shows that the task is not solved by luck.

Run either with the policy runner, for example::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type isaaclab_arena_environments.spatial_memory.policies.SpatialMemoryOraclePolicy \\
        --num_episodes 10 spatial_memory_mug_past_door
"""

from __future__ import annotations

import gymnasium as gym
import torch
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_environments.spatial_memory.controller import ToolPoseTracker
from isaaclab_arena_environments.spatial_memory.runtime import get_runtime
from isaaclab_arena_environments.spatial_memory.task_logic import PlanMode


@dataclass
class SpatialMemoryOraclePolicyCfg(PolicyCfg):
    """Configure the spatial-memory oracle controller."""


@dataclass
class SpatialMemoryBlindPolicyCfg(PolicyCfg):
    """Configure the spatial-memory blind controller."""


class _SpatialMemoryScriptedPolicy:
    """Shared implementation of the oracle and blind controllers."""

    mode: str = PlanMode.ORACLE

    def _init_scripted(self) -> None:
        self._tracker: ToolPoseTracker | None = None
        self._needs_plan: torch.Tensor | None = None

    def _scripted_action(self, env: gym.Env) -> torch.Tensor:
        base_env = env.unwrapped
        runtime = get_runtime(base_env)
        logic = runtime.logic
        adapter = logic.embodiment
        if self._tracker is None:
            self._tracker = ToolPoseTracker(
                base_env.num_envs, base_env.device, base_env.step_dt, adapter.rest_joint_pose
            )
            self._needs_plan = torch.ones(base_env.num_envs, dtype=torch.bool)
        for env_id in torch.nonzero(self._needs_plan).flatten().tolist():
            self._tracker.set_plan(env_id, logic.plan(base_env, runtime, env_id, self.mode))
        self._needs_plan[:] = False
        T_E_tool = logic.tool_pose_w(base_env).clone()
        T_E_tool[:, :3] -= base_env.scene.env_origins
        joint_pos, joint_limits = adapter.arm_joint_state(base_env)
        item_positions_e = {name: base_env.arena_world.get_pose_e(name)[:, :3] for name in logic.item_names()}
        joint_targets, gripper = self._tracker.compute_joint_targets(
            T_E_tool,
            adapter.tool_jacobian(base_env),
            joint_pos,
            joint_limits,
            logic.plan_gates(base_env, runtime),
            item_positions_e,
        )
        return adapter.make_action(base_env, joint_targets, gripper)

    def _scripted_reset(self, env_ids: torch.Tensor | None) -> None:
        if self._needs_plan is None:
            return
        if env_ids is None:
            self._needs_plan[:] = True
        else:
            self._needs_plan[env_ids.cpu()] = True


@register_policy
class SpatialMemoryOraclePolicy(_SpatialMemoryScriptedPolicy, PolicyBase[SpatialMemoryOraclePolicyCfg]):
    """Privileged waypoint controller that uses the hidden geometry."""

    name = "spatial_memory_oracle"
    mode = PlanMode.ORACLE

    def __init__(self, config: SpatialMemoryOraclePolicyCfg | None = None):
        super().__init__(config or SpatialMemoryOraclePolicyCfg())
        self._init_scripted()

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        return self._scripted_action(env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._scripted_reset(env_ids)


@register_policy
class SpatialMemoryBlindPolicy(_SpatialMemoryScriptedPolicy, PolicyBase[SpatialMemoryBlindPolicyCfg]):
    """Waypoint controller that ignores hidden geometry (straight-line motions or nominal guesses)."""

    name = "spatial_memory_blind"
    mode = PlanMode.BLIND

    def __init__(self, config: SpatialMemoryBlindPolicyCfg | None = None):
        super().__init__(config or SpatialMemoryBlindPolicyCfg())
        self._init_scripted()

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        return self._scripted_action(env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._scripted_reset(env_ids)
