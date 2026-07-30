# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.variation_recorder import VariationRecorder


@dataclass
class _DeformableVisualMeshSync:
    """Render-only proxy meshes for one PhysX deformable asset."""

    asset_name: str
    asset: Any
    proxy_prims: list[Any]
    proxy_translate_ops: list[Any]


class IsaacLabArenaManagerBasedRLEnv(ManagerBasedRLEnv):
    """Arena extension to ManagerBasedRLEnv that adds additional Arena-specific functionality."""

    cfg: IsaacLabArenaManagerBasedRLEnvCfg

    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        variation_recorder: VariationRecorder | None = None,
        arena_scene_assets: dict[str, Asset] | None = None,
        **kwargs,
    ):
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(
            num_envs=cfg.scene.num_envs, device=cfg.sim.device
        )
        self._variation_recorder = variation_recorder
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        self._arena_scene_assets = dict(arena_scene_assets or {})
        # Per-env count of completed episodes; advanced in ``_reset_idx``.
        self._episode_counts: dict[int, int] = {}
        # The initial reset touches every env before any episode has run; skip it.
        self._first_reset = True
        self._deformable_visual_mesh_syncs: list[_DeformableVisualMeshSync] = []
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._setup_deformable_visual_mesh_syncs()
        self._sync_deformable_visual_meshes()

    @property
    def variation_recorder(self) -> VariationRecorder | None:
        """The recorder of variation samples, or ``None`` if the env was not built with one."""
        if self._variation_recorder is None:
            print(
                "[WARNING] variation_recorder is None; no variation samples were recorded. "
                "Build the env through ArenaEnvBuilder to record variations."
            )
        return self._variation_recorder

    @property
    def object_initial_rest_pose_recorder(self) -> ObjectInitialRestPoseRecorder:
        """The recorder of initial object rest poses. Used when object_settled predicate is enabled by task progress tracking."""
        return self._object_initial_rest_pose_recorder

    @property
    def arena_scene_assets(self) -> dict[str, Asset]:
        """Original Arena scene assets keyed by name, used for runtime geometry queries."""
        return self._arena_scene_assets

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    def load_managers(self) -> None:
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the index of the current episode in ``env_id``."""
        return self._episode_counts.get(env_id, 0)

    def _advance_episode_indices(self, env_ids: Sequence[int]) -> None:
        """Advance the per-env episode counter for each episode in ``env_ids``."""
        for env_id in env_ids:
            env_id = int(env_id)
            self._episode_counts[env_id] = self._episode_counts.get(env_id, 0) + 1

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # The initial reset touches every env before any episode has run; nothing to record or count.
        if self._first_reset:
            self._first_reset = False
            super()._reset_idx(env_ids)
            return
        # Runs recorder before super() so the just-finished episode is still intact.
        self.episode_recorder_manager.record_pre_reset(env_ids)
        # Advance before super() so reset-mode variation draws are tagged with the episode they begin.
        self._advance_episode_indices(env_ids)
        super()._reset_idx(env_ids)

    def reset(self, *args, **kwargs):
        """Reset the env, then refresh any render-only deformable visual proxies."""
        result = super().reset(*args, **kwargs)
        self._sync_deformable_visual_meshes()
        return result

    def step(self, action):
        """Step the env, then refresh any render-only deformable visual proxies."""
        result = super().step(action)
        self._sync_deformable_visual_meshes()
        return result

    def render(self, recompute: bool = False):
        """Render after refreshing any render-only deformable visual proxies."""
        self._sync_deformable_visual_meshes()
        return super().render(recompute=recompute)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()

    def _setup_deformable_visual_mesh_syncs(self) -> None:
        """Create render-only proxy meshes for PhysX deformables when requested."""
        if not getattr(self.cfg, "sync_deformable_visual_meshes_from_sim", False):
            return
        if not getattr(self.scene, "physics_backend", "").startswith("physx"):
            return
        if self.render_mode != "rgb_array" and not self.sim.has_gui and not self.sim.is_rendering:
            return

        for asset_name, asset in self.scene.deformable_objects.items():
            proxy_prims = self._create_deformable_visual_proxies(asset)
            if proxy_prims:
                self._deformable_visual_mesh_syncs.append(
                    _DeformableVisualMeshSync(
                        asset_name=asset_name,
                        asset=asset,
                        proxy_prims=[proxy_prim for proxy_prim, _translate_op in proxy_prims],
                        proxy_translate_ops=[translate_op for _proxy_prim, translate_op in proxy_prims],
                    )
                )

    def _create_deformable_visual_proxies(self, asset) -> list[tuple[Any, Any]]:
        """Create one unbound render proxy mesh per deformable instance."""
        from pxr import Gf, UsdGeom, UsdShade

        proxies = []
        for root_prim, source_mesh_prim in self._deformable_visual_mesh_prims(asset):
            stage = root_prim.GetStage()
            proxy_path = root_prim.GetPath().AppendChild("arena_visual_proxy")
            proxy_mesh = UsdGeom.Mesh.Define(stage, proxy_path)
            source_mesh = UsdGeom.Mesh(source_mesh_prim)

            proxy_mesh.GetPointsAttr().Set(source_mesh.GetPointsAttr().Get())
            proxy_mesh.GetFaceVertexIndicesAttr().Set(source_mesh.GetFaceVertexIndicesAttr().Get())
            proxy_mesh.GetFaceVertexCountsAttr().Set(source_mesh.GetFaceVertexCountsAttr().Get())
            if source_mesh.GetSubdivisionSchemeAttr().HasAuthoredValueOpinion():
                proxy_mesh.GetSubdivisionSchemeAttr().Set(source_mesh.GetSubdivisionSchemeAttr().Get())

            material, _binding = UsdShade.MaterialBindingAPI(source_mesh_prim).ComputeBoundMaterial()
            if material:
                UsdShade.MaterialBindingAPI.Apply(proxy_mesh.GetPrim()).Bind(material)

            xformable = UsdGeom.Xformable(proxy_mesh.GetPrim())
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

            UsdGeom.Imageable(source_mesh_prim).MakeInvisible()
            UsdGeom.Imageable(proxy_mesh.GetPrim()).MakeVisible()
            proxies.append((proxy_mesh.GetPrim(), translate_op))
        return proxies

    def _deformable_visual_mesh_prims(self, asset) -> list[tuple[Any, Any]]:
        """Return ``(root_prim, visual_mesh_prim)`` pairs for a deformable asset."""
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom

        pairs = []
        for root_prim, _root_expr in sim_utils.resolve_matching_prims_from_source(asset.cfg.prim_path):
            meshes = sim_utils.get_all_matching_child_prims(
                root_prim.GetPath().pathString,
                lambda prim: (
                    prim.GetTypeName() == "Mesh"
                    and not prim.GetName().startswith("arena_visual_proxy")
                    and UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.guide
                ),
            )
            if not meshes:
                meshes = sim_utils.get_all_matching_child_prims(
                    root_prim.GetPath().pathString,
                    lambda prim: prim.GetTypeName() == "Mesh" and not prim.GetName().startswith("arena_visual_proxy"),
                )
            if len(meshes) != 1:
                continue
            pairs.append((root_prim, meshes[0]))
        return pairs

    def _sync_deformable_visual_meshes(self) -> None:
        """Copy PhysX deformable nodal positions to render-only proxy meshes."""
        if not self._deformable_visual_mesh_syncs:
            return

        import numpy as np

        import warp as wp
        from pxr import Gf, UsdGeom, Vt

        for sync in self._deformable_visual_mesh_syncs:
            asset = sync.asset
            raw_nodal_pos = asset.root_view.get_simulation_nodal_positions()
            nodal_pos = (
                wp.to_torch(raw_nodal_pos)
                .reshape((asset.num_instances, asset.max_sim_vertices_per_body, 3))
                .detach()
                .cpu()
                .numpy()
            )
            xform_cache = UsdGeom.XformCache()
            for env_id, (proxy_prim, translate_op) in enumerate(zip(sync.proxy_prims, sync.proxy_translate_ops)):
                if env_id >= nodal_pos.shape[0]:
                    break
                parent_prim = proxy_prim.GetParent()
                parent_inverse_transform = xform_cache.GetLocalToWorldTransform(parent_prim).GetInverse()
                centroid_w = nodal_pos[env_id].mean(axis=0)
                centroid_local = parent_inverse_transform.Transform(
                    Gf.Vec3d(float(centroid_w[0]), float(centroid_w[1]), float(centroid_w[2]))
                )
                translate_op.Set(centroid_local)
                local_points = []
                for point in nodal_pos[env_id]:
                    local_point = parent_inverse_transform.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                    local_points.append(
                        Gf.Vec3f(
                            float(local_point[0] - centroid_local[0]),
                            float(local_point[1] - centroid_local[1]),
                            float(local_point[2] - centroid_local[2]),
                        )
                    )

                proxy_mesh = UsdGeom.Mesh(proxy_prim)
                proxy_mesh.GetPointsAttr().Set(Vt.Vec3fArray(local_points))

                points_array = np.array([[point[0], point[1], point[2]] for point in local_points], dtype=np.float32)
                min_point = points_array.min(axis=0)
                max_point = points_array.max(axis=0)
                proxy_mesh.GetExtentAttr().Set(
                    Vt.Vec3fArray([
                        Gf.Vec3f(float(min_point[0]), float(min_point[1]), float(min_point[2])),
                        Gf.Vec3f(float(max_point[0]), float(max_point[1]), float(max_point[2])),
                    ])
                )
