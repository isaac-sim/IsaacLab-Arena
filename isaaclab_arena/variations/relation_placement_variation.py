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

from isaaclab_arena.relations.placement_poses import get_scene_root_poses_from_layout
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv
from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


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
    resample_on_reset: bool = True
    """Whether live placement draws a fresh pooled layout on reset.

    Recorded replay always follows the replay scheduler and ignores this setting.
    """


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
