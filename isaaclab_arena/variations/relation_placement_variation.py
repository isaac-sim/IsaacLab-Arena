# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene-level relation-placement variation."""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING, Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.relations.placement_poses import get_scene_root_poses_from_layout
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv
from isaaclab_arena.variations.relation_placement_sampler import (
    PlacementPoolSampler,
    PlacementSamplerCfg,
    validate_placement_samples,
)
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


@configclass
class RelationPlacementVariationCfg(VariationBaseCfg):
    """Configuration for the automatically enabled placement variation."""

    enabled: bool = True
    sampler_cfg: PlacementSamplerCfg = field(default_factory=PlacementSamplerCfg)
    resample_on_reset: bool = True
    """Whether live placement draws a fresh pooled layout on reset.

    Recorded replay always follows the replay scheduler and ignores this setting.
    """


class RelationPlacementHandle:
    """Opaque event parameter retaining the placement sampler."""

    __slots__ = ("sampler",)

    def __init__(self, sampler: PlacementPoolSampler) -> None:
        self.sampler = sampler

    def __deepcopy__(self, memo: dict[int, object]) -> RelationPlacementHandle:
        memo[id(self)] = self
        return self


class RelationPlacementVariation(RunTimeVariationBase):
    """Coordinate complete relation-placement samples across scene roots."""

    HOST_NAME = "scene"
    NAME = "relation_placement"
    EVENT_NAME = "scene_relation_placement"
    reset_priority = 100

    def __init__(
        self,
        sampler: PlacementPoolSampler,
        *,
        num_envs: int,
        cfg: RelationPlacementVariationCfg | None = None,
    ) -> None:
        self.name = self.NAME
        self._sampler = sampler
        self._sample_listeners = []
        self._replay_sampler = None
        self.cfg = cfg if cfg is not None else RelationPlacementVariationCfg()
        self._num_envs = num_envs
        self._recorded_replay_samples: list[Any] | None = None
        self._prepared = False
        assert self.cfg.enabled, "Relation placement is automatically enabled when relations are configured"

    @property
    def placement_pool(self) -> PooledObjectPlacer:
        """Return the active live pool used for relation solving."""
        assert self.has_live_pool, "Recorded placement replay has no active live placement pool"
        assert self._sampler.placement_pool is not None
        return self._sampler.placement_pool

    @property
    def has_live_pool(self) -> bool:
        """Whether draws currently come from a live relation-solving pool."""
        return self._sampler.placement_pool is not None and self._recorded_replay_samples is None

    @property
    def can_supply_samples(self) -> bool:
        """Whether live solving or recorded replay can supply placement samples."""
        return self._sampler.placement_pool is not None or self._recorded_replay_samples is not None

    @property
    def last_results(self) -> dict[int, PlacementResult]:
        """Return solver results applied by the latest live reset."""
        return dict(self._sampler.last_results)

    def apply_cfg(self, cfg: RelationPlacementVariationCfg) -> None:
        """Apply Hydra configuration without replacing builder-owned sampler state."""
        assert cfg.enabled, "scene.relation_placement.enabled=false is unsupported; disable relation solving instead"
        self.cfg = cfg

    def _prepare_at_build_time(self) -> None:
        """Prepare live or replayed construction poses before scene materialisation."""
        if self._prepared:
            return
        if self._recorded_replay_samples is not None:
            _seed_spawn_config_from_replay(
                self._recorded_replay_samples,
                self._sampler.replay_assets,
                self._num_envs,
            )
        else:
            assert (
                self._sampler.placement_pool is not None
            ), "Relation placement has neither recorded samples nor a live pool"
            anchor_assets = set(get_anchor_objects(self._sampler.assets))
            _validate_no_conflicting_pose_reset_events(self._sampler.assets, anchor_assets)
            layouts = self._sampler.prepare_live(self._num_envs, self.cfg.resample_on_reset)
            _seed_spawn_config_from_layouts(self._sampler.assets, anchor_assets, layouts)
            if self._sampler.placement_pool.had_fallbacks:
                print(
                    "Warning: Relation placement pool accepted best-loss fallback layouts "
                    "that failed strict placement validation."
                )
        if self._sampler.placement_pool is not None:
            self._sampler.placement_pool.release_build_time_dependencies()
        self._prepared = True

    def on_replay_samples_bound(self, samples: list[Any] | None) -> None:
        """Retain validated placement rows needed to seed construction poses."""
        self._recorded_replay_samples = samples

    def validate_replay_samples(self, samples: list[Any]) -> None:
        """Validate complete poses against the current scene roots."""
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.relations.placement_asset import get_scene_root_owners
        from isaaclab_arena.relations.placement_layouts import validate_root_reset_for_placement_replay
        from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

        validate_placement_samples(samples)
        assert not any(isinstance(asset, RigidObjectSet) for asset in self._sampler.replay_assets), (
            "Recorded placement replay does not support RigidObjectSet; "
            "use homogeneous assets or omit scene.relation_placement from the variation samples."
        )
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
        handle = RelationPlacementHandle(self._sampler)
        return (
            self.EVENT_NAME,
            EventTermCfg(func=apply_relation_placement_sample, mode="reset", params={"placement": handle}),
        )


def apply_relation_placement_sample(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    placement: RelationPlacementHandle,
) -> None:
    """Draw, record, and apply one complete placement per reset environment."""
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners

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
    pose_keys = rows[0]["poses"].keys()
    poses = {
        key: torch.stack(
            [Pose.from_dict(row["poses"][key]).to_tensor(device=env.device) for row in rows],
        )
        for key in pose_keys
    }
    asset_poses: dict[PlaceableAsset, dict[str, torch.Tensor]] = {}
    root_owners = get_scene_root_owners(placement.sampler.replay_assets)
    for key, pose in poses.items():
        asset_poses.setdefault(root_owners[key], {})[key] = pose
    for asset, owned_poses in asset_poses.items():
        asset.write_scene_root_poses_to_sim(env, env_ids, owned_poses)


def _seed_spawn_config_from_layouts(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
    layouts: list[PlacementResult],
) -> None:
    """Seed every scene root from environment-indexed solved layouts without asset reset events."""
    poses_by_asset: dict[PlaceableAsset, dict[str, list[Pose]]] = {}
    for layout in layouts:
        root_poses_by_asset = get_scene_root_poses_from_layout(assets, layout, anchor_assets)
        for asset, root_poses in root_poses_by_asset.items():
            poses_per_root = poses_by_asset.setdefault(asset, {key: [] for key in root_poses})
            for key, pose in root_poses.items():
                poses_per_root[key].append(pose)
    for asset, poses_per_root in poses_by_asset.items():
        asset.set_initial_scene_root_poses({key: PosePerEnv(poses=poses) for key, poses in poses_per_root.items()})


def _seed_spawn_config_from_replay(
    samples: list[dict[str, Any]],
    replay_assets: list[PlaceableAsset],
    num_envs: int,
) -> None:
    """Seed construction roots from replay rows without creating asset reset events."""
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners

    owners = get_scene_root_owners(replay_assets)
    poses_by_asset: dict[PlaceableAsset, dict[str, PosePerEnv]] = {}
    for scene_key in samples[0]["poses"]:
        per_env_poses = []
        for env_id in range(num_envs):
            pose = Pose.from_dict(samples[env_id % len(samples)]["poses"][scene_key])
            assert pose is not None
            per_env_poses.append(pose)
        poses_by_asset.setdefault(owners[scene_key], {})[scene_key] = PosePerEnv(per_env_poses)
    for asset, poses in poses_by_asset.items():
        asset.set_initial_scene_root_poses(poses)


def _validate_no_conflicting_pose_reset_events(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
) -> None:
    """Reject conflicting explicit pose-reset events on relation-solved assets."""
    for asset in assets:
        assert not (asset not in anchor_assets and asset.has_pose_reset_event()), (
            f"Non-anchor asset '{asset.name}' has an explicit pose-reset event. "
            "Relational solving should not be combined with explicit setting of "
            "poses on non-anchor assets."
        )


def _placement_pose_keys_from_assets(assets: list[PlaceableAsset]) -> set[str]:
    anchor_assets = set(get_anchor_objects(assets))
    return {scene_key for asset in assets if asset not in anchor_assets for scene_key in asset.get_scene_root_keys()}


def get_relation_placement_variation(env: Any) -> RelationPlacementVariation | None:
    """Return the environment's scene relation-placement variation, when configured."""
    base = env.unwrapped
    variation = base.scene_variations.get(RelationPlacementVariation.NAME)
    if variation is None:
        return None
    assert isinstance(variation, RelationPlacementVariation)
    return variation


def get_relation_placement_pool(env: Any) -> PooledObjectPlacer:
    """Return the environment's active live relation-placement pool."""
    variation = get_relation_placement_variation(env)
    assert variation is not None, "Environment has no relation-placement variation"
    assert variation.has_live_pool, "Relation placement is using recorded replay, not live sampling"
    return variation.placement_pool
