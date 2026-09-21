# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered Newton environments for the gear-insertion tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from ..registration import register_environment

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_EASY_SCENE_SPEC = Path(__file__).with_name("gear_easy.yaml")
_EASY_PAIR_SCENE_SPEC = Path(__file__).with_name("gear_easy_pair.yaml")
_MEDIUM_TRAIN_SCENE_SPEC = Path(__file__).with_name("gear_medium_train.yaml")


def _configure_gear_mesh_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Match AUTOLab's 60 Hz x 16-substep contact-rich gear regime."""
    from isaaclab_newton.physics import NewtonCollisionPipelineCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    from .physics import disable_mjwarp_sensors

    physics = disable_mjwarp_sensors(deepcopy(ArenaPhysicsCfg().newton))
    physics.collision_decimation = 1
    physics.solver_cfg.update_data_interval = 1
    physics.default_shape_cfg.ke = 60_000.0
    physics.default_shape_cfg.kd = 500.0
    env_cfg.sim.dt = 1.0 / 60.0
    env_cfg.decimation = 1
    physics.num_substeps = 16
    # This contact-rich model exceeds a 48 GiB card during Warp graph
    # instantiation at the upstream buffers; eager Warp keeps the same solver.
    physics.use_cuda_graph = False
    physics.solver_cfg.njmax = 32768
    physics.solver_cfg.nconmax = 16384
    physics.solver_cfg.use_mujoco_contacts = False
    physics.solver_cfg.enable_multiccd = True
    physics.solver_cfg.iterations = 100
    physics.solver_cfg.ls_iterations = 50
    physics.solver_cfg.cone = "elliptic"
    physics.solver_cfg.impratio = 10.0
    # AUTOLab's native-MuJoCo path uses enable_multiccd=True to retain a
    # multi-point manifold for meshing convex pieces.  On Newton's collision
    # path, preserving generated contacts instead of reducing them is the
    # corresponding control.
    physics.collision_cfg = NewtonCollisionPipelineCfg(
        reduce_contacts=False,
        rigid_contact_max=32768,
        max_triangle_pairs=1_000_000,
    )
    physics.default_shape_cfg.gap = 5.0e-5
    env_cfg.sim.physics = physics
    return env_cfg


@dataclass
class GearMeshNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the registered gear-insertion environment."""

    enable_cameras: bool = False
    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False
    episode_length_s: float | None = None


@dataclass
class GearInsertionEasyNewtonEnvironmentCfg(GearMeshNewtonEnvironmentCfg):
    """Configure the registered easy gear-insertion environment."""


@dataclass
class GearMeshPairNewtonEnvironmentCfg(GearMeshNewtonEnvironmentCfg):
    """Configure the two-station generated Gear Mesh family."""

    layout_names: list[str] = field(default_factory=list)


@dataclass
class GearMeshTrainNewtonEnvironmentCfg(GearMeshNewtonEnvironmentCfg):
    """Configure the three-station generated Gear Mesh family."""

    layout_names: list[str] = field(default_factory=list)


class GearMeshNewtonEnvironment(ArenaEnvironmentFactory[GearMeshNewtonEnvironmentCfg]):
    """Build gear insertion from its graph and task-owned Newton profile."""

    _legacy_argparse_cfg_type = GearMeshNewtonEnvironmentCfg
    scene_spec: Path
    env_cfg_callback = staticmethod(_configure_gear_mesh_physics)

    def build(self, cfg: GearMeshNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena_environments.isaac_cap import register_components

        register_components()
        spec = ArenaEnvGraphSpec.from_yaml(str(self.scene_spec))
        arena_env = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
        arena_env.embodiment.camera_config.use_overhead_profile("gear")
        arena_env.embodiment.set_use_tiled_cameras(cfg.use_tiled_cameras)
        arena_env.embodiment.set_use_instanceable_meshes(cfg.use_instanceable_meshes)
        from .mesh_instancing import configure_scene_mesh_instancing

        configure_scene_mesh_instancing(arena_env.scene, cfg.use_instanceable_meshes)
        if cfg.episode_length_s is not None:
            if cfg.episode_length_s <= 0:
                raise ValueError("episode_length_s must be positive")
            arena_env.task.episode_length_s = cfg.episode_length_s
        arena_env.env_cfg_callback = partial(self.env_cfg_callback)
        return arena_env


@register_environment(cfg_type=GearInsertionEasyNewtonEnvironmentCfg)
class GearInsertionEasyNewtonEnvironment(GearMeshNewtonEnvironment):
    """Build AUTOLab gearmesh-easy-single-01."""

    name = "vabar_contact_rich_insertion_v2__gear_easy"
    _legacy_argparse_cfg_type = GearInsertionEasyNewtonEnvironmentCfg
    scene_spec = _EASY_SCENE_SPEC
    env_cfg_callback = staticmethod(_configure_gear_mesh_physics)

    def build(self, cfg: GearInsertionEasyNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        arena_env = super().build(cfg)
        from .task.variation import GearFamilyVariation

        board = arena_env.scene.assets["board"]
        gear = arena_env.scene.assets["gear_a"]
        board.add_variation(GearFamilyVariation(board, gear, arena_env.task))
        return arena_env


class _GearMeshLayoutNewtonEnvironment(GearMeshNewtonEnvironment):
    """Build one exact-row generated Gear Mesh family."""

    family: str
    gear_names: tuple[str, ...]
    env_cfg_callback = staticmethod(_configure_gear_mesh_physics)

    def build(self, cfg: GearMeshNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        arena_env = super().build(cfg)
        from .task.parallel_layouts import configure_parallel_layouts
        from .task.variation import GearLayoutVariation, GearLayoutVariationCfg

        board = arena_env.scene.assets["board"]
        gears = tuple(arena_env.scene.assets[name] for name in self.gear_names)
        if not cfg.layout_names:
            board.add_variation(
                GearLayoutVariation(
                    board,
                    gears,
                    arena_env.task,
                    GearLayoutVariationCfg(family=self.family),
                )
            )
        physics_callback = self.env_cfg_callback

        def configure(env_cfg):
            return configure_parallel_layouts(
                physics_callback(env_cfg),
                family=self.family,
                gear_names=self.gear_names,
                layout_names=cfg.layout_names,
            )

        arena_env.env_cfg_callback = configure
        return arena_env


@register_environment(cfg_type=GearMeshPairNewtonEnvironmentCfg)
class GearMeshPairNewtonEnvironment(_GearMeshLayoutNewtonEnvironment):
    """Build AUTOLab's generated gearmesh-easy-pair family."""

    name = "vabar_contact_rich_insertion_v2__gear_easy_pair"
    _legacy_argparse_cfg_type = GearMeshPairNewtonEnvironmentCfg
    scene_spec = _EASY_PAIR_SCENE_SPEC
    family = "pair"
    gear_names = ("gear_a", "gear_b")


@register_environment(cfg_type=GearMeshTrainNewtonEnvironmentCfg)
class GearMeshTrainNewtonEnvironment(_GearMeshLayoutNewtonEnvironment):
    """Build AUTOLab's generated gearmesh-medium-train family."""

    name = "vabar_contact_rich_insertion_v2__gear_medium_train"
    _legacy_argparse_cfg_type = GearMeshTrainNewtonEnvironmentCfg
    scene_spec = _MEDIUM_TRAIN_SCENE_SPEC
    family = "train"
    gear_names = ("gear_a", "gear_b", "gear_c")
