# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal B=1 Franka Arena environment for the CAP barrier smoke.

Import this module only after ``SimulationApp`` has started. Its construction path
uses Arena's normal scene/embodiment builder rather than a standalone Isaac Lab task.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .joint_mapping import PANDA_ARM_JOINTS, make_franka_joint_mapping


class FrankaSimulationAdapter:
    """Expose the narrow simulator surface consumed by ``ArenaLockstepManager``."""

    def __init__(self, environment: Any):
        import torch

        self._torch = torch
        self._environment = environment
        self._unwrapped = environment.unwrapped
        if self._unwrapped.num_envs != 1:
            raise RuntimeError(f"CAP barrier smoke requires B=1, got {self._unwrapped.num_envs}")
        self._robot = self._unwrapped.scene["robot"]
        self._joint_names = tuple(self._robot.joint_names)
        self.joint_mapping = make_franka_joint_mapping(self._joint_names)

        action_manager = self._unwrapped.action_manager
        active_terms = tuple(action_manager.active_terms)
        term_dimensions = tuple(action_manager.action_term_dim)
        if active_terms != ("arm_action", "gripper_action") or term_dimensions != (7, 1):
            raise RuntimeError(
                "CAP Franka profile requires action terms arm_action[7], gripper_action[1], "
                f"got {list(zip(active_terms, term_dimensions, strict=True))}"
            )
        arm_term = action_manager.get_term("arm_action")
        self.joint_mapping.assert_action_order(arm_term._joint_names)
        self._arm_slice = slice(0, 7)
        self._gripper_slice = slice(7, 8)
        self.physics_step_count = 0
        self.reset_count = 0

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    @staticmethod
    def _as_tensor(value):
        return value.torch if hasattr(value, "torch") else value

    def read_joint_state(self) -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
        position = self._as_tensor(self._robot.data.joint_pos)[0].detach().cpu().tolist()
        velocity = self._as_tensor(self._robot.data.joint_vel)[0].detach().cpu().tolist()
        # This is the best available smoke-level effort sample. Its parity with a
        # physical joint-state effort remains a World State contract question.
        effort = self._as_tensor(self._robot.data.applied_torque)[0].detach().cpu().tolist()
        return position, velocity, effort

    def arm_positions(self) -> tuple[float, ...]:
        positions, _, _ = self.read_joint_state()
        return self.joint_mapping.to_abi_order(positions)

    def step_position_targets(self, positions_in_abi_order: Sequence[float]) -> None:
        if len(positions_in_abi_order) != 7:
            raise ValueError(f"expected seven Franka arm targets, got {len(positions_in_abi_order)}")
        action = self._torch.zeros(
            (1, self._unwrapped.action_manager.total_action_dim),
            device=self._unwrapped.device,
            dtype=self._as_tensor(self._robot.data.joint_pos).dtype,
        )
        action[:, self._arm_slice] = self._torch.as_tensor(
            positions_in_abi_order,
            device=self._unwrapped.device,
            dtype=action.dtype,
        )
        action[:, self._gripper_slice] = 1.0
        self._environment.step(action)
        self.physics_step_count += 1

    def reset_without_physics_step(self) -> None:
        # ManagerBasedEnv.reset writes reset state and performs sim.forward(); it
        # does not advance physics, which is the required CR-20 fence behavior.
        self._environment.reset()
        self.reset_count += 1

    def close(self) -> None:
        self._environment.close()


def make_cap_franka_environment(*, device: str = "cuda:0") -> FrankaSimulationAdapter:
    """Build the fixed arena_droid_b1 smoke profile after Kit startup."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    registry = AssetRegistry()
    embodiment = registry.get_asset_by_name("franka_joint_pos")()
    arm_action = embodiment.action_config.arm_action
    arm_action.joint_names = list(PANDA_ARM_JOINTS)
    arm_action.preserve_order = True
    arm_action.scale = 1.0
    arm_action.offset = 0.0
    arm_action.use_default_offset = False

    ground_plane = registry.get_asset_by_name("ground_plane")()
    light = registry.get_asset_by_name("light")()
    scene = Scene(assets=[ground_plane, light])

    def configure_profile(cfg):
        cfg.sim.dt = 0.005
        cfg.decimation = 1
        return cfg

    description = IsaacLabArenaEnvironment(
        name="CAP-Barrier-Franka-B1-v0",
        scene=scene,
        embodiment=embodiment,
        task=NoTask(),
        env_cfg_callback=configure_profile,
    )
    builder = ArenaEnvBuilder(
        description,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device),
    )
    environment, cfg = builder.make_registered_and_return_cfg()
    if cfg.sim.dt != 0.005 or cfg.decimation != 1:
        environment.close()
        raise RuntimeError(f"CAP profile timing mismatch: dt={cfg.sim.dt}, decimation={cfg.decimation}")
    return FrankaSimulationAdapter(environment)
