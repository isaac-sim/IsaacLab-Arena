# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene-level relation-placement variation."""

from __future__ import annotations

import math
import torch
from dataclasses import field
from numbers import Real
from typing import TYPE_CHECKING, Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.relations.placement_events import get_pose_from_layout, write_scene_poses_to_sim
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


SCENE_VARIATION_HOST = "scene"
RELATION_PLACEMENT_VARIATION_NAME = "relation_placement"
RELATION_PLACEMENT_EVENT_NAME = "scene_relation_placement"


@configclass
class PlacementSamplerCfg(SamplerBaseCfg):
    """Hydra marker for the live placement sampler owned by the builder."""

    def build(self) -> SamplerBase:
        raise RuntimeError("PlacementSamplerCfg requires builder-owned placement state")


@configclass
class RelationPlacementVariationCfg(VariationBaseCfg):
    """Configuration for the automatically enabled placement variation."""

    enabled: bool = True
    sampler_cfg: PlacementSamplerCfg = field(default_factory=PlacementSamplerCfg)


class PlacementPoolSampler(SamplerBase):
    """Draw complete scene-pose rows from a placement pool or fixed per-env layouts."""

    def __init__(
        self,
        assets: list[PlaceableAsset],
        placement_pool: PooledObjectPlacer | None,
        fixed_results: dict[int, PlacementResult] | None = None,
        replay_assets: list[PlaceableAsset] | None = None,
    ) -> None:
        super().__init__()
        self.assets = assets
        self.placement_pool = placement_pool
        self.fixed_results = fixed_results
        self.replay_assets = replay_assets or assets
        self.last_results: dict[int, PlacementResult] = {}
        self._next_layout_id = 0
        self._fixed_rows = (
            {env_id: self._serialize_result(result) for env_id, result in fixed_results.items()}
            if fixed_results is not None
            else None
        )

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
            if self.fixed_results is None:
                results = self.placement_pool.sample_for_envs(env_id_list)
                rows = [self._serialize_result(results[env_id]) for env_id in env_id_list]
            else:
                results = {env_id: self.fixed_results[env_id] for env_id in env_id_list}
                assert self._fixed_rows is not None
                rows = [self._fixed_rows[env_id] for env_id in env_id_list]
            self.last_results = results
            if self.fixed_results is None:
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
        anchor_assets = set(get_anchor_objects(self.assets))
        for asset in self.assets:
            if asset in anchor_assets:
                continue
            layout_pose = get_pose_from_layout(asset, result)
            for scene_key, pose in asset.layout_pose_to_scene_writes(layout_pose):
                assert scene_key not in poses, f"Duplicate relation-placement scene root: {scene_key!r}"
                poses[scene_key] = pose.to_dict()
        row: dict[str, Any] = {
            "layout_id": f"layout_{self._next_layout_id:06d}",
            "source": "relation_solver",
            "poses": poses,
        }
        self._next_layout_id += 1
        return row


class RelationPlacementHandle:
    """Opaque event parameter retaining the live placement sampler."""

    __slots__ = ("sampler", "write_live_samples")

    def __init__(self, sampler: PlacementPoolSampler, write_live_samples: bool) -> None:
        self.sampler = sampler
        self.write_live_samples = write_live_samples

    def __deepcopy__(self, memo: dict[int, object]) -> RelationPlacementHandle:
        memo[id(self)] = self
        return self


class RelationPlacementVariation(RunTimeVariationBase):
    """Coordinate complete relation-placement samples across scene roots."""

    reset_priority = 100

    def __init__(
        self,
        sampler: PlacementPoolSampler,
        *,
        write_live_samples: bool,
        cfg: RelationPlacementVariationCfg | None = None,
    ) -> None:
        self.name = RELATION_PLACEMENT_VARIATION_NAME
        self._sampler = sampler
        self._sample_listeners = []
        self._replay_sampler = None
        self.cfg = cfg if cfg is not None else RelationPlacementVariationCfg()
        self._write_live_samples = write_live_samples
        assert self.cfg.enabled, "Relation placement is automatically enabled when relations are configured"

    @property
    def placement_pool(self) -> PooledObjectPlacer:
        """Return the live pool used for relation solving."""
        assert self._sampler.placement_pool is not None, "Placement replay has no live placement pool"
        return self._sampler.placement_pool

    @property
    def has_live_pool(self) -> bool:
        """Whether this variation owns a live relation-solving pool."""
        return self._sampler.placement_pool is not None

    @property
    def last_results(self) -> dict[int, PlacementResult]:
        """Return solver results applied by the latest live reset."""
        return dict(self._sampler.last_results)

    def apply_cfg(self, cfg: RelationPlacementVariationCfg) -> None:
        """Apply Hydra configuration without replacing builder-owned sampler state."""
        assert cfg.enabled, "scene.relation_placement.enabled=false is unsupported; disable relation solving instead"
        self.cfg = cfg

    def validate_replay_samples(self, samples: list[Any]) -> None:
        """Validate complete poses against the current scene roots."""
        from isaaclab_arena.relations.placement_asset import get_scene_root_owners
        from isaaclab_arena.relations.placement_layouts import validate_root_reset_for_placement_replay
        from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

        validate_placement_samples(samples)
        required_keys = _placement_pose_keys_from_assets(self._sampler.assets)
        allowed_keys = {scene_key for asset in self._sampler.replay_assets for scene_key in asset.get_scene_root_keys()}
        owners = get_scene_root_owners(self._sampler.replay_assets)
        for sample in samples:
            pose_keys = set(sample["poses"])
            assert required_keys <= pose_keys and pose_keys <= allowed_keys, (
                "Placement replay scene keys differ from the current environment; "
                f"missing={sorted(required_keys - pose_keys)}, unknown={sorted(pose_keys - allowed_keys)}"
            )
            for asset in self._sampler.replay_assets:
                owned_keys = set(asset.get_scene_root_keys())
                if pose_keys.intersection(owned_keys):
                    assert (
                        owned_keys <= pose_keys
                    ), f"Placement replay is missing roots owned by '{asset.name}': {sorted(owned_keys - pose_keys)}"
        selected_assets = {owners[key] for key in samples[0]["poses"]}
        # TODO(qianl) [variation-replay-consistency]: Ensure replayed scene roots refer to the same concrete objects that produced the recording.
        validate_root_reset_for_placement_replay(list(selected_assets))
        for asset in selected_assets:
            assert (
                get_relation(asset, RandomAroundSolution) is None
            ), f"Placement replay object '{asset.name}' cannot randomize on reset"

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        handle = RelationPlacementHandle(self._sampler, self._write_live_samples)
        return (
            RELATION_PLACEMENT_EVENT_NAME,
            EventTermCfg(func=apply_relation_placement_sample, mode="reset", params={"placement": handle}),
        )


def apply_relation_placement_sample(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    placement: RelationPlacementHandle,
) -> None:
    """Draw, record, and when required apply one complete placement per reset environment."""
    if env_ids is None or len(env_ids) == 0:
        return
    placement_pool = placement.sampler.placement_pool
    if placement_pool is not None:
        assert placement_pool.num_envs == env.scene.env_origins.shape[0], (
            f"Placement pool has {placement_pool.num_envs} envs, "
            f"but scene has {env.scene.env_origins.shape[0]} env origins."
        )
    else:
        assert (
            placement.sampler.replays_recorded_samples
        ), "Relation placement has neither replay samples nor a live pool"
    rows = placement.sampler.sample(len(env_ids), env_ids)
    if not placement.write_live_samples and not placement.sampler.replays_recorded_samples:
        return
    pose_keys = rows[0]["poses"].keys()
    poses = {
        key: torch.stack(
            [Pose.from_dict(row["poses"][key]).to_tensor(device=env.device) for row in rows],
        )
        for key in pose_keys
    }
    write_scene_poses_to_sim(env, env_ids, poses)


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


def _placement_pose_keys_from_assets(assets: list[PlaceableAsset]) -> set[str]:
    anchor_assets = set(get_anchor_objects(assets))
    return {scene_key for asset in assets if asset not in anchor_assets for scene_key in asset.get_scene_root_keys()}


def get_relation_placement_variation(env: Any) -> RelationPlacementVariation | None:
    """Return the environment's scene relation-placement variation, when configured."""
    base = env.unwrapped
    variation = base.scene_variations.get(RELATION_PLACEMENT_VARIATION_NAME)
    if variation is None:
        return None
    assert isinstance(variation, RelationPlacementVariation)
    return variation
