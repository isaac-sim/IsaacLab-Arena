# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build Isaac Lab MDP2 environment configs from Arena environments."""

from __future__ import annotations

import torch
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.utils.physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab_experimental import mdp2
    from isaaclab_experimental.envs import MDP2RLEnvCfg

    from isaaclab_arena.relations.runtime.placement_sampler import PlacementSampler

LAYOUT_EVENT_NAME = "relation_placement"
"""Name of the reset event that draws relation-solved layouts."""


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
            parts.append(_make_layout_task_cfg(placement_sampler, layout_pool_size))

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


def _make_layout_task_cfg(placement_sampler: PlacementSampler, pool_size: int) -> mdp2.TaskCfg:
    """Return a partial task with a reset event that writes relation-solved layouts of the movable assets.

    Args:
        placement_sampler: The live placement sampler, whose pool holds solved layouts.
        pool_size: Number of layouts drawn from the sampler's pool to fill the event's pool.

    Returns:
        A partial task config with the event :data:`LAYOUT_EVENT_NAME`.
    """
    from isaaclab_experimental import mdp2

    from isaaclab_arena.mdp2 import terms
    from isaaclab_arena.relations.runtime.placement_poses import get_scene_root_poses_from_layout

    scene_keys = [key for asset in placement_sampler.write_assets for key in asset.get_scene_root_keys()]
    rows = []
    for result in placement_sampler.placement_pool.sample_with_replacement(pool_size):
        poses_by_asset = get_scene_root_poses_from_layout(placement_sampler.assets, result)
        poses = {key: pose for root_poses in poses_by_asset.values() for key, pose in root_poses.items()}
        rows.append([value for key in scene_keys for value in (*poses[key].position_xyz, *poses[key].rotation_xyzw)])
    outputs = {}
    for k, key in enumerate(scene_keys):
        entity = mdp2.Entity(key)
        outputs[f"pose_{k}"] = entity.root_link_pose_w
        outputs[f"vel_{k}"] = entity.root_com_vel_w
    event = mdp2.Event(
        terms.make_layout_pool_term(torch.tensor(rows, dtype=torch.float32)),
        inputs={"env_origins": mdp2.Entity("scene").env_origins},
        outputs=outputs,
    )
    return mdp2.TaskCfg(events={LAYOUT_EVENT_NAME: event})
