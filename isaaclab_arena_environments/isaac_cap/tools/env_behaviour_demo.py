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
