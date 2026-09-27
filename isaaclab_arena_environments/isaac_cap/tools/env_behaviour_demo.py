# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


class EnvBehaviourDemo(ABC):
    """Run environment-specific validation cycles in a live simulation app."""

    label = "env_behaviour_demo"

    def __init__(
        self,
        simulation_app: SimulationAppContext,
        arena_environment: IsaacLabArenaEnvironment,
        builder_cfg: ArenaEnvBuilderCfg,
        *,
        real_time: bool = True,
        visualizer_cfg: Any | None = None,
    ) -> None:
        """Configure shared environment construction, lifecycle, and step pacing.

        Args:
            simulation_app: Active Arena simulation application context.
            arena_environment: Composed Arena environment to instantiate.
            builder_cfg: Configuration for building the stepable environment.
            real_time: Whether to pace environment steps using the environment step period. Defaults to true.
            visualizer_cfg: Optional default simulator visualizer configuration.
        """
        self.simulation_app = simulation_app
        self.arena_environment = arena_environment
        self.builder_cfg = builder_cfg
        self.real_time = real_time
        self.visualizer_cfg = visualizer_cfg
        self._env: gym.Env | None = None
        self._rate_limiter: Any | None = None

    @property
    def env(self) -> gym.Env:
        """Return the wrapped environment while the demo is running."""
        assert self._env is not None, "The validation environment has not been created."
        return self._env

    @property
    def base_env(self) -> Any:
        """Return the unwrapped Arena environment while the demo is running."""
        return self.env.unwrapped

    def make_env(self) -> gym.Env:
        """Build and return the wrapped, stepable environment used by this demo."""
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        builder = ArenaEnvBuilder(self.arena_environment, self.builder_cfg)
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        if self.visualizer_cfg is not None:
            env_cfg.sim.default_visualizer_cfg = self.visualizer_cfg
        return builder.make_registered(env_cfg, env_kwargs)

    def setup_demo(self) -> None:
        """Perform environment-specific setup after the initial reset."""

    @abstractmethod
    def run_cycle(self, cycle: int) -> None:
        """Run one environment-specific validation cycle."""

    def is_running(self) -> bool:
        """Return whether the simulation application can continue stepping."""
        return self.simulation_app.is_running() and not self.simulation_app.is_exiting()

    def step(self, action: Any) -> tuple[Any, ...]:
        """Step the environment once and apply the configured pacing."""
        if not self.is_running():
            raise KeyboardInterrupt
        transition = self.env.step(action)
        if self._rate_limiter is not None:
            self._rate_limiter.sleep()
        return transition

    def render(self) -> None:
        """Render the current state without advancing the environment."""
        if not self.is_running():
            raise KeyboardInterrupt
        self.base_env.sim.render()
        if self._rate_limiter is not None:
            self._rate_limiter.sleep()

    def run_demo(self, cycles: int = 0) -> None:
        """Build the environment and run cycles until completion or simulation shutdown.

        Args:
            cycles: Number of cycles to run. Zero runs until the simulation closes.
        """
        from isaaclab_arena.utils.rate_limiter import RateLimiter

        assert cycles >= 0, "cycles must be non-negative; zero means repeat until the simulation closes."
        assert self._env is None, "An EnvBehaviourDemo instance can only be run once."
        self._env = self.make_env()
        self._rate_limiter = RateLimiter(self.base_env.step_dt) if self.real_time else None

        try:
            self.env.reset()
            self.setup_demo()
            cycle = 1
            while self.is_running() and (cycles == 0 or cycle <= cycles):
                self.run_cycle(cycle)
                cycle += 1
        except KeyboardInterrupt:
            print(f"\n[{self.label}] exiting", flush=True)
        finally:
            self.env.close()


class DifferentialIKEnvBehaviourDemo(EnvBehaviourDemo):
    """Share two-environment FR3 differential-IK motion helpers."""

    max_translation_per_step_m: float
    move_to_max_steps: int
    position_tolerance_m: float

    def setup_differential_ik(self, expected_num_envs: int = 2) -> None:
        """Resolve the differential-IK action term and robot end-effector."""
        import torch

        self.torch = torch
        self.num_envs = self.base_env.num_envs
        assert self.num_envs == expected_num_envs, f"Expected {expected_num_envs} environments, got {self.num_envs}."
        action_manager = self.base_env.action_manager
        assert action_manager.active_terms == [
            "arm_action",
            "gripper_action",
        ], f"Unexpected action terms: {action_manager.active_terms}."
        assert action_manager.total_action_dim == 7, (
            "The validation demo requires six relative IK commands and one gripper command; "
            f"got {action_manager.total_action_dim} actions."
        )
        self.arm_action = action_manager.get_term("arm_action")
        self.robot = self.base_env.scene["robot"]
        body_ids, _ = self.robot.find_bodies("robotiq_base")
        assert len(body_ids) == 1, f"Expected one robotiq_base body, got {body_ids}."
        self.ee_body_id = int(body_ids[0])

    def _ee_position(self):
        return self.robot.data.body_pos_w.torch[:, self.ee_body_id].clone()

    def _ik_action(self, translation_delta_w=None, *, gripper_closed: bool):
        """Build a relative-IK action, converting world translation into robot-base coordinates."""
        import isaaclab.utils.math as math_utils

        action = self.torch.zeros(
            (self.num_envs, self.base_env.action_manager.total_action_dim),
            device=self.base_env.device,
        )
        if translation_delta_w is not None:
            delta_b = math_utils.quat_apply_inverse(
                self.robot.data.root_quat_w.torch,
                translation_delta_w,
            )
            distance = self.torch.linalg.vector_norm(delta_b, dim=-1, keepdim=True)
            fraction = self.torch.clamp(
                self.max_translation_per_step_m / distance.clamp_min(1.0e-9),
                max=1.0,
            )
            action[:, :3] = delta_b * fraction / self.arm_action._scale[:, :3]
        action[:, -1] = float(gripper_closed)
        return action

    def _step(self, action):
        """Step once and return the terminated-or-truncated mask."""
        _, _, terminated, truncated, _ = self.step(action)
        return terminated | truncated

    def _hold_ik(self, steps: int, *, gripper_closed: bool) -> bool:
        """Hold the current IK pose and report whether any environment ended."""
        return any(bool(self._step(self._ik_action(gripper_closed=gripper_closed)).any().item()) for _ in range(steps))

    def _move_to(
        self,
        target_position_w,
        *,
        gripper_closed: bool,
        label: str,
        position_tolerance_m: float | None = None,
    ) -> bool:
        """Drive toward one Cartesian position and report whether any environment ended."""
        tolerance = self.position_tolerance_m if position_tolerance_m is None else position_tolerance_m
        for _ in range(self.move_to_max_steps):
            error_w = target_position_w - self._ee_position()
            errors_m = self.torch.linalg.vector_norm(error_w, dim=-1)
            if bool((errors_m <= tolerance).all().item()):
                return self._hold_ik(10, gripper_closed=gripper_closed)
            if bool(self._step(self._ik_action(error_w, gripper_closed=gripper_closed)).any().item()):
                return True
        errors_m = self.torch.linalg.vector_norm(target_position_w - self._ee_position(), dim=-1)
        if bool((errors_m <= tolerance).all().item()):
            return self._hold_ik(10, gripper_closed=gripper_closed)
        raise RuntimeError(f"Timed out during {label}; maximum end-effector position error is {errors_m.max():.3f} m.")
