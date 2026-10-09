# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sampling and validation for relation-placement variation rows."""

from __future__ import annotations

import math
import torch
from numbers import Real
from typing import TYPE_CHECKING, Any

from isaaclab.utils.configclass import configclass

from isaaclab_arena.relations.placement_poses import get_scene_root_poses_from_layout
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


@configclass
class PlacementSamplerCfg(SamplerBaseCfg):
    """Hydra marker for the live placement sampler owned by the builder."""

    def build(self) -> SamplerBase:
        raise RuntimeError("PlacementSamplerCfg requires builder-owned placement state")


class PlacementPoolSampler(SamplerBase):
    """Draw complete scene-pose rows from a placement pool or fixed per-env layouts."""

    def __init__(
        self,
        assets: list[PlaceableAsset],
        placement_pool: PooledObjectPlacer | None,
        replay_assets: list[PlaceableAsset] | None = None,
    ) -> None:
        super().__init__()
        self.assets = assets
        self.placement_pool = placement_pool
        self.replay_assets = replay_assets or assets
        self.last_results: dict[int, PlacementResult] = {}
        self._next_layout_id = 0
        self._fixed_results: dict[int, PlacementResult] | None = None
        self._fixed_rows: dict[int, dict[str, Any]] | None = None

    def prepare_live(self, num_envs: int, resample_on_reset: bool) -> list[PlacementResult]:
        """Draw construction layouts and retain them when resets should stay fixed."""
        assert self.placement_pool is not None, "Live relation placement requires a placement pool"
        if resample_on_reset:
            [construction_layout] = self.placement_pool.sample_with_replacement(1)
            return [construction_layout] * num_envs

        layouts = self.placement_pool.sample_with_replacement(num_envs)
        self._fixed_results = {env_id: layout for env_id, layout in enumerate(layouts)}
        self._fixed_rows = {env_id: self._serialize_result(result) for env_id, result in self._fixed_results.items()}
        return layouts

    @property
    def replays_recorded_samples(self) -> bool:
        """Whether draws currently come from the variation replay scheduler."""
        return self._replay_sampler is not None

    def sample(self, num_samples: int, env_ids: torch.Tensor) -> list[dict[str, Any]]:
        """Return one serializable complete placement row per requested environment."""
        assert env_ids is not None, "Relation placement requires explicit environment ids"
        env_id_list = [int(env_id) for env_id in env_ids.tolist()]
        assert num_samples == len(env_id_list), "Placement sample count must match env_ids"
        replay_rows = self._get_replay_samples(num_samples, env_ids)
        if replay_rows is not None:
            rows = replay_rows
            self.last_results = {}
        else:
            assert self.placement_pool is not None, "Live relation placement requires a placement pool"
            if self._fixed_results is None:
                results = self.placement_pool.sample_for_envs(env_id_list)
                rows = [self._serialize_result(results[env_id]) for env_id in env_id_list]
            else:
                results = {env_id: self._fixed_results[env_id] for env_id in env_id_list}
                assert self._fixed_rows is not None
                rows = [self._fixed_rows[env_id] for env_id in env_id_list]
            self.last_results = results
            if self._fixed_results is None:
                for env_id, result in results.items():
                    if not result.success:
                        print(
                            "Warning: Writing best-loss fallback placement for "
                            f"env {env_id}; failed checks: "
                            f"{result.validation_results.get_failed_validation_check_names}."
                        )
        validate_placement_samples(rows)
        self._notify(rows, env_ids)
        return rows

    def _serialize_result(self, result: PlacementResult) -> dict[str, Any]:
        poses: dict[str, dict[str, list[float]]] = {}
        for root_poses in get_scene_root_poses_from_layout(self.assets, result).values():
            for scene_key, pose in root_poses.items():
                poses[scene_key] = pose.to_dict()
        row: dict[str, Any] = {
            "layout_id": f"layout_{self._next_layout_id:06d}",
            "source": "relation_solver",
            "poses": poses,
        }
        self._next_layout_id += 1
        return row


def validate_placement_samples(samples: list[Any]) -> None:
    """Require serializable complete placement rows."""
    assert samples, "Placement replay requires at least one sample"
    expected_keys: set[str] | None = None
    for sample in samples:
        assert isinstance(sample, dict), "Placement samples must be mappings"
        assert isinstance(sample.get("layout_id"), str) and sample["layout_id"], "Placement layout_id must be nonempty"
        assert isinstance(sample.get("source"), str) and sample["source"], "Placement source must be nonempty"
        poses = sample.get("poses")
        assert isinstance(poses, dict) and poses, "Placement poses must be a nonempty mapping"
        if expected_keys is None:
            expected_keys = set(poses)
        assert set(poses) == expected_keys, "Every placement sample must contain the same scene roots"
        for key, value in poses.items():
            assert isinstance(key, str) and key, "Placement scene keys must be nonempty strings"
            assert isinstance(value, dict) and set(value) == {
                "position_xyz",
                "rotation_xyzw",
            }, f"Placement pose for {key!r} requires position_xyz and rotation_xyzw"
            for field_name, size in (("position_xyz", 3), ("rotation_xyzw", 4)):
                components = value[field_name]
                assert isinstance(components, list) and len(components) == size
                assert all(
                    isinstance(component, Real) and not isinstance(component, bool) and math.isfinite(component)
                    for component in components
                )
            Pose.from_dict(value)
            assert math.isclose(
                sum(component * component for component in value["rotation_xyzw"]),
                1.0,
                abs_tol=1e-4,
            ), f"Placement pose for {key!r} requires a unit quaternion"
        identities = sample.get("assets")
        if identities is not None:
            assert (
                isinstance(identities, dict) and identities.keys() == poses.keys()
            ), "Placement assets must map every pose key"
            assert all(isinstance(value, str) and value for value in identities.values())
