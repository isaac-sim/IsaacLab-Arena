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

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_prim_tree import load_usd_physics_roots

if TYPE_CHECKING:
    from pxr import Usd


class Background(Object):
    """
    Encapsulates the background scene for a environment.
    """

    def __init__(
        self,
        name: str,
        usd_path: str,
        object_min_z: float,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        reset_nested_physics: bool = False,
        **kwargs,
    ):
        self.reset_nested_physics = reset_nested_physics
        super().__init__(
            name=name,
            usd_path=usd_path,
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
        """Return a USD spawner that materializes nested instance-proxy physics."""
        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        if self.reset_nested_physics:
            assert isinstance(cfg, UsdFileCfg), "Nested background physics requires a USD file spawner"
            cfg = cfg.copy()
            cfg.func = _spawn_from_usd_with_resettable_nested_physics
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


@clone
def _spawn_from_usd_with_resettable_nested_physics(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD background and materialize instance-proxy physics subtrees."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # PhysX tensor views cannot bind dynamic instance proxies. Materialize only
    # instanceable subtrees containing physics so reset views can control them.
    _deinstance_nested_physics(prim)
    return prim
