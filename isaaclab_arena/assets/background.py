# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.string import string_to_callable

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.physics_config import UsdFileCfgPrimPhysicsWrapper
from isaaclab_arena.assets.sources.base import UsdSource
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.prim_tree import load_usd_physics_roots

if TYPE_CHECKING:
    from pxr import Usd


class Background(Object):
    """
    Encapsulates the background scene for a environment.
    """

    def __init__(
        self,
        name: str,
        object_min_z: float,
        usd_path: str | None = None,
        usd_source: UsdSource | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        reset_nested_physics: bool = True,
        **kwargs,
    ):
        self.reset_nested_physics = reset_nested_physics
        super().__init__(
            name=name,
            usd_path=usd_path,
            usd_source=usd_source,
            initial_pose=initial_pose,
            prim_path=prim_path,
            # The background root is non-physical, though nested prims may have physics.
            object_type=ObjectType.BASE,
            **kwargs,
        )
        # We use this to define reset terms for when objects are dropped.
        # NOTE(alexmillane, 2025.09.19): This is a global z height. If you shift the
        # background, by using initial_pose, this height doesn't shift with it.
        # TODO(alexmillane, 2025.09.19): Make this value relative to the background
        # prim origin.
        self.object_min_z = object_min_z

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Return the background's spawn config.

        The returned spawner deactivates the physics scenes authored in the background USD. With
        reset_nested_physics, it also de-instances subtrees containing rigid bodies or articulations and
        replaces any configured USD spawn function. Without reset_nested_physics, a config with a custom
        spawn function is returned unchanged, so the physics scenes in its USD stay active.
        """
        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        if self.reset_nested_physics:
            assert isinstance(cfg, UsdFileCfg), "Nested background physics requires a USD file spawner"
        elif not _spawns_with_standard_usd_spawner(cfg):
            return cfg
        cfg = cfg.copy()
        spawn_func = (
            _spawn_from_usd_with_resettable_nested_physics
            if self.reset_nested_physics
            else _spawn_from_usd_without_physics_scenes
        )
        if isinstance(cfg, UsdFileCfgPrimPhysicsWrapper):
            # The per-prim physics spawner runs its wrapped USD spawner before applying the edits.
            cfg.usd_spawn_func = spawn_func
        else:
            cfg.func = spawn_func
        return cfg

    def get_nested_physics_prim_paths(
        self,
        referenced_prim_paths: dict[str, ObjectType] | None = None,
    ) -> dict[str, ObjectType]:
        """Return nested physics root paths not owned by object references.

        Args:
            referenced_prim_paths: Runtime paths represented by explicit object references.

        Returns:
            Runtime path templates mapped to their physics object types.
        """
        if not self.reset_nested_physics:
            return {}

        referenced_relative_paths: dict[str, ObjectType] = {}
        prim_path_prefix = f"{self.prim_path}/"
        for prim_path, object_type in (referenced_prim_paths or {}).items():
            assert prim_path.startswith(
                prim_path_prefix
            ), f"Referenced prim '{prim_path}' is outside background '{self.prim_path}'"
            referenced_relative_paths[prim_path.removeprefix(prim_path_prefix)] = object_type

        physics_roots = load_usd_physics_roots(
            self.usd_path,
            referenced_prim_paths=referenced_relative_paths,
        )
        return {
            f"{self.prim_path}/{relative_path}": object_type for relative_path, object_type in physics_roots.items()
        }

    def get_viewer_cfg(self) -> ViewerCfg | None:
        """Return a custom viewer camera framing for this background, or None to auto-frame."""


def _deinstance_nested_physics(prim: Usd.Prim) -> None:
    """De-instance subtrees that contribute dynamic physics prims."""
    from pxr import Usd, UsdPhysics

    instance_roots: dict[str, Usd.Prim] = {}
    root_path = prim.GetPath()
    for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not (candidate.HasAPI(UsdPhysics.RigidBodyAPI) or candidate.HasAPI(UsdPhysics.ArticulationRootAPI)):
            continue
        ancestor = candidate
        while ancestor.IsValid() and ancestor.GetPath().HasPrefix(root_path):
            if ancestor.IsInstance():
                instance_roots[str(ancestor.GetPath())] = ancestor
                break
            if ancestor == prim:
                break
            ancestor = ancestor.GetParent()

    for path, instance_root in sorted(instance_roots.items()):
        deinstanced = instance_root.SetInstanceable(False)
        assert deinstanced, f"Failed to de-instance nested physics subtree '{path}'"


def _spawns_with_standard_usd_spawner(cfg) -> bool:
    """Return whether the config spawns with Isaac Lab's spawn_from_usd, directly or under per-prim physics."""
    if type(cfg) is UsdFileCfg:
        spawn_func = cfg.func
    elif type(cfg) is UsdFileCfgPrimPhysicsWrapper:
        spawn_func = cfg.usd_spawn_func
    else:
        return False
    if isinstance(spawn_func, str):
        spawn_func = string_to_callable(str(spawn_func))
    return spawn_func is spawn_from_usd


def _deactivate_physics_scenes(prim: Usd.Prim) -> None:
    """Deactivate the physics scenes under the spawned background prim, except those inside instanced subtrees."""
    from pxr import Usd, UsdPhysics

    # A background USD saved with its own scene adds one PhysX scene per environment,
    # and PhysX stalls in the first simulation step when there are many.
    # Collect before deactivating; editing the stage can invalidate a live traversal.
    physics_scenes = [candidate for candidate in Usd.PrimRange(prim) if candidate.IsA(UsdPhysics.Scene)]
    for physics_scene in physics_scenes:
        physics_scene.SetActive(False)


@clone
def _spawn_from_usd_without_physics_scenes(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD background with its own physics scenes deactivated."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    _deactivate_physics_scenes(prim)
    return prim


@clone
def _spawn_from_usd_with_resettable_nested_physics(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD background, de-instance its physics subtrees, and deactivate its own physics scenes."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # PhysX tensor views cannot bind dynamic instance proxies. Materialize only
    # instanceable subtrees containing physics so reset views can control them.
    _deinstance_nested_physics(prim)
    # After de-instancing, so scenes inside those subtrees are editable too.
    _deactivate_physics_scenes(prim)
    return prim
