# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build Isaac Lab MDP2 environment configs from Arena environments."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.utils.physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab_experimental.envs import MDP2RLEnvCfg


class ArenaMDP2EnvBuilder(ArenaEnvBuilder):
    """Build an Isaac Lab MDP2 environment config from an Arena environment.

    The scene comes from the scene, embodiment, and task as for a manager-based config. Their MDP comes from their
    partial MDP2 task configs, ``get_mdp2_task_cfg()``, which ``mdp2.compose`` merges entry by entry. Relation
    placement becomes a reset event that draws solved layouts from a pool on the GPU.
    """

    def build_mdp2_env_cfg(
        self, camera_width: int = 320, camera_height: int = 180, layout_pool_size: int = 256
    ) -> MDP2RLEnvCfg:
        """Build the MDP2 environment config of the Arena environment.

        Args:
            camera_width: Rendered width of every camera in pixels.
            camera_height: Rendered height of every camera in pixels.
            layout_pool_size: Number of relation-solved layouts that reset worlds draw from.

        Returns:
            The environment config, for ``MDP2RLEnv``, after the environment's ``env_cfg_callback`` tunes it.
        """
        import isaaclab.sim as sim_utils
        from isaaclab_experimental import mdp2
        from isaaclab_experimental.envs import MDP2RLEnvCfg, ResetStep

        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg
        from isaaclab_arena.mdp2.placement import make_layout_task_cfg

        assert self.resolved_physics_backend is PhysicsBackend.NEWTON, "MDP2 environments run on Newton"
        arena_env = self.arena_env
        embodiment, task = arena_env.embodiment, arena_env.task
        assert embodiment is not None and task is not None, "An MDP2 environment needs an embodiment and a task"

        placement_sampler = self._prepare_relation_placement(None)
        embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
        task.configure_for_embodiment(embodiment)
        scene_cfg = self._build_scene_cfg(embodiment, task)
        _set_cameras_to_newton_rendering(scene_cfg, camera_width, camera_height)

        parts = [part for part in (embodiment.get_mdp2_task_cfg(), task.get_mdp2_task_cfg()) if part is not None]
        if placement_sampler is not None:
            # Last, so that it places the assets after the reset events that set their defaults.
            parts.append(make_layout_task_cfg(placement_sampler, layout_pool_size))

        env_cfg = MDP2RLEnvCfg(
            sim=sim_utils.SimulationCfg(dt=1 / 120, render_interval=8, physics=ArenaPhysicsCfg().newton),
            scene=scene_cfg,
            task=mdp2.compose(*parts),
            decimation=8,
            seed=self.cfg.seed,
            # Render again after resets, so that the first images of an episode show its new layout.
            reset_step=[ResetStep.EVENTS, ResetStep.COMMIT, ResetStep.RENDER, ResetStep.OBSERVE],
        )
        if arena_env.env_cfg_callback is not None:
            env_cfg = arena_env.env_cfg_callback(env_cfg)
        return env_cfg


def _set_cameras_to_newton_rendering(scene_cfg: Any, width: int, height: int) -> None:
    """Render every camera of the scene with Newton's Warp renderer, once per control step, at the given size."""
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    for field in fields(scene_cfg):
        camera = getattr(scene_cfg, field.name)
        if not isinstance(camera, CameraCfg):
            continue
        camera.renderer_cfg = NewtonWarpRendererCfg()
        camera.update_period = 0.0
        # Cameras on moving links follow them for every render; this is free for fixed ones.
        camera.update_latest_camera_pose = True
        camera.width, camera.height = width, height
