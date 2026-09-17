# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_arena.utils.physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab_arena.assets.teleop_device_base import TeleopDeviceBase
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import TaskBase


class IsaacLabArenaEnvironment:
    """Describes an environment in IsaacLab Arena."""

    def __init__(
        self,
        name: str,
        scene: Scene,
        embodiment: EmbodimentBase | None = None,
        task: TaskBase | None = None,
        teleop_device: TeleopDeviceBase | None = None,
        env_cfg_callback: (
            Callable[[IsaacLabArenaManagerBasedRLEnvCfg], IsaacLabArenaManagerBasedRLEnvCfg] | None
        ) = None,
        rl_framework_entry_point: str | None = None,
        rl_policy_cfg: str | None = None,
        episode_recorder_terms: dict[str, EpisodeRecorderTermCfg] | None = None,
        placer_params: ObjectPlacerParams | None = None,
        default_physics_backend: PhysicsBackend = PhysicsBackend.PHYSX,
        embodiments: list[EmbodimentBase] | None = None,
    ):
        """
        Args:
            name: The name of the environment.
            scene: The scene to use in the environment.
            embodiment: The embodiment to use in the environment.
            task: The task to use in the environment.
            teleop_device: The teleop device to use in the environment.
            env_cfg_callback: A callback that tunes the environment configuration after the
                resolved physics backend is materialized. It must not change the backend type.
            rl_framework_entry_point: Gym kwargs key under which the RL policy config is
                registered. This is an IsaacLab convention: each supported RL framework has a
                fixed key that its training scripts look up via ``load_cfg_from_registry``.
                Common values: ``"rsl_rl_cfg_entry_point"``, ``"skrl_cfg_entry_point"``,
                ``"sb3_cfg_entry_point"``, ``"rl_games_cfg_entry_point"``. Required when
                ``rl_policy_cfg`` is set.
            rl_policy_cfg: Import path to the RL policy config class, e.g.
                ``"my_module:RLPolicyCfg"``.
            episode_recorder_terms: Additional per-episode recorder terms to record alongside the
                built-in ones, keyed by name.
            placer_params: Object placement configuration. When None, default
                ObjectPlacerParams are used.
            default_physics_backend: Default physics backend when ``--presets`` is omitted.
            embodiments: Robots in action-tensor order. Mutually exclusive with ``embodiment``.
        """
        self.name = name
        self.scene = scene
        assert embodiment is None or embodiments is None, "Specify embodiment or embodiments, not both"
        self.embodiments = list(embodiments) if embodiments is not None else ([embodiment] if embodiment else [])
        self.validate_embodiments()
        self.task = task
        self.teleop_device = teleop_device
        self.env_cfg_callback = env_cfg_callback
        if (rl_framework_entry_point is None) != (rl_policy_cfg is None):
            raise ValueError("rl_framework_entry_point and rl_policy_cfg must both be set or both be None.")
        self.rl_framework_entry_point = rl_framework_entry_point
        self.rl_policy_cfg = rl_policy_cfg
        self.episode_recorder_terms = episode_recorder_terms or {}
        self.placer_params = placer_params
        self.default_physics_backend = PhysicsBackend(default_physics_backend)

    def validate_embodiments(self) -> None:
        """Require distinct scene keys, including at most one unkeyed robot."""
        assert (
            sum(robot.instance_key is None for robot in self.embodiments) <= 1
        ), "At most one embodiment may be unkeyed"
        keys = [embodiment.get_scene_key() for embodiment in self.embodiments]
        assert len(keys) == len(set(keys)), "Embodiment scene keys must be unique; at most one robot may be unkeyed"

    @property
    def embodiment(self) -> EmbodimentBase | None:
        """Return the sole robot, or None when the environment has no robots."""
        assert len(self.embodiments) <= 1, "Use embodiments for an environment with several robots"
        return self.embodiments[0] if self.embodiments else None

    @embodiment.setter
    def embodiment(self, embodiment: EmbodimentBase | None) -> None:
        """Replace the robot list through the single-robot convenience interface."""
        self.embodiments = [embodiment] if embodiment is not None else []
