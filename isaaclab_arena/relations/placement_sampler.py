# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler-level recording and replay for relation placement."""

from __future__ import annotations

import json
import math
import torch
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.relations.placement_poses import get_scene_root_poses_from_layout
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.sampler_base import SamplerBase

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


@dataclass(frozen=True)
class PlacementSample:
    """One placement draw expressed as native scene-root poses."""

    layout_id: str
    poses: dict[str, Pose]

    def __post_init__(self) -> None:
        assert self.layout_id, "Placement layout_id must be nonempty"
        _validate_native_poses(self.poses)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible representation used for recording and replay."""
        return {
            "layout_id": self.layout_id,
            "poses": {scene_key: pose.to_dict() for scene_key, pose in self.poses.items()},
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> PlacementSample:
        """Decode one validated replay record into native scene-root poses."""
        return cls(
            layout_id=record["layout_id"],
            poses={scene_key: Pose.from_dict(pose) for scene_key, pose in record["poses"].items()},
        )


class PlacementSampler(SamplerBase):
    """Draw complete native scene-root samples from live, fixed, or recorded layouts."""

    def __init__(
        self,
        assets: list[PlaceableAsset],
        placement_pool: PooledObjectPlacer | None,
        replay_assets: list[PlaceableAsset] | None = None,
    ) -> None:
        super().__init__()
        self.assets = assets
        self.placement_pool = placement_pool
        self.replay_assets = assets if replay_assets is None else replay_assets
        self.last_results: dict[int, PlacementResult] = {}
        self._next_layout_id = 0
        self._fixed_placements: dict[int, tuple[PlacementResult, PlacementSample]] | None = None

    @property
    def replays_recorded_samples(self) -> bool:
        """Whether draws currently come from the episode replay scheduler."""
        return self._replay_sampler is not None

    def prepare_live(self, num_envs: int, resolve_on_reset: bool) -> list[PlacementResult]:
        """Draw construction layouts and retain them when resets should stay fixed."""
        assert self.placement_pool is not None, "Live relation placement requires a placement pool"
        if resolve_on_reset:
            [construction_layout] = self.placement_pool.sample_with_replacement(1)
            return [construction_layout] * num_envs

        layouts = self.placement_pool.sample_with_replacement(num_envs)
        self._fixed_placements = {
            env_id: (layout, self._sample_from_result(layout)) for env_id, layout in enumerate(layouts)
        }
        return layouts

    def sample(self, num_samples: int, env_ids: torch.Tensor) -> list[PlacementSample]:
        """Return one complete placement sample per requested environment."""
        assert env_ids is not None, "Relation placement requires explicit environment ids"
        env_id_list = [int(env_id) for env_id in env_ids.tolist()]
        assert num_samples == len(env_id_list), "Placement sample count must match env_ids"
        replay_rows = self._get_replay_samples(num_samples, env_ids)
        if replay_rows is not None:
            samples = [PlacementSample.from_record(row) for row in replay_rows]
            self.last_results = {}
        else:
            assert self.placement_pool is not None, "Live relation placement requires a placement pool"
            if self._fixed_placements is None:
                results = self.placement_pool.sample_for_envs(env_id_list)
                samples = [self._sample_from_result(results[env_id]) for env_id in env_id_list]
            else:
                results = {}
                samples = []
                for env_id in env_id_list:
                    result, sample = self._fixed_placements[env_id]
                    results[env_id] = result
                    samples.append(sample)
            self.last_results = results
            if self._fixed_placements is None:
                for env_id, result in results.items():
                    if not result.success:
                        print(
                            "Warning: Writing best-loss fallback placement for "
                            f"env {env_id}; failed checks: "
                            f"{result.validation_results.get_failed_validation_check_names}."
                        )
        self._notify(samples, env_ids)
        return samples

    def _sample_from_result(self, result: PlacementResult) -> PlacementSample:
        poses_by_asset = get_scene_root_poses_from_layout(self.assets, result)
        sample = PlacementSample(
            layout_id=f"layout_{self._next_layout_id:06d}",
            poses={scene_key: pose for root_poses in poses_by_asset.values() for scene_key, pose in root_poses.items()},
        )
        self._next_layout_id += 1
        return sample


def serialize_placement_samples(samples: list[PlacementSample]) -> list[dict[str, Any]]:
    """Convert native placement samples to JSON-compatible records."""
    assert all(isinstance(sample, PlacementSample) for sample in samples)
    return [sample.to_record() for sample in samples]


def validate_placement_samples(samples: list[Any]) -> None:
    """Require complete, finite, consistently keyed placement records."""
    assert samples, "Placement replay requires at least one sample"
    expected_keys: set[str] | None = None
    for sample in samples:
        assert isinstance(sample, dict), "Placement samples must be mappings"
        assert isinstance(sample.get("layout_id"), str) and sample["layout_id"], "Placement layout_id must be nonempty"
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
        PlacementSample.from_record(sample)


def placement_samples_from_pose_columns(poses: dict[str, list[Pose]]) -> list[PlacementSample]:
    """Convert scene-root pose columns into ordered placement samples."""
    assert poses, "Placement samples must contain scene roots"
    counts = {len(root_poses) for root_poses in poses.values()}
    assert (
        len(counts) == 1 and next(iter(counts)) > 0
    ), "All placement scene roots must have the same nonzero number of poses"
    return [
        PlacementSample(
            layout_id=f"layout_{index:06d}",
            poses={scene_key: root_poses[index] for scene_key, root_poses in poses.items()},
        )
        for index in range(next(iter(counts)))
    ]


def placement_samples_to_pose_columns(samples: list[PlacementSample]) -> dict[str, list[Pose]]:
    """Convert ordered placement samples into scene-root pose columns."""
    _validate_native_samples(samples)
    return {scene_key: [sample.poses[scene_key] for sample in samples] for scene_key in samples[0].poses}


def read_placement_samples(path: str | Path) -> list[PlacementSample]:
    """Read placement samples from offline or episode-result JSONL records."""
    from isaaclab_arena.recording.episode_results import read_episode_records

    samples: list[PlacementSample] = []
    for record_index, record in enumerate(read_episode_records(path), start=1):
        try:
            placement = record.get("placement")
            if placement is None:
                placement = record["variations"]["scene.relation_placement"]
            validate_placement_samples([placement])
            samples.append(PlacementSample.from_record(placement))
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise AssertionError(f"{path}, record {record_index}: {error}") from error
    try:
        _validate_native_samples(samples)
    except AssertionError as error:
        raise AssertionError(f"{path}: {error}") from error
    return samples


def write_placement_samples(
    path: str | Path,
    samples: list[PlacementSample],
    validation: list[dict] | None = None,
) -> None:
    """Write placement samples and optional validation reports as top-level JSONL rows."""
    _validate_native_samples(samples)
    assert validation is None or len(validation) == len(samples), "One validation report is required per sample"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for index, sample in enumerate(samples):
            placement = sample.to_record()
            if validation is not None:
                placement["validation"] = validation[index]
            stream.write(json.dumps({"placement": placement}, allow_nan=False) + "\n")


def validate_placement_sample_assets(samples: list[PlacementSample], assets: list[PlaceableAsset]) -> None:
    """Require concrete root ownership and complete coverage of every selected asset."""
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners
    from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

    _validate_native_samples(samples)
    assert not any(
        isinstance(asset, RigidObjectSet) for asset in assets
    ), "Recorded placement requires concrete assets, not object sets"
    pose_keys = set(samples[0].poses)
    owners = get_scene_root_owners(assets)
    unknown = pose_keys - owners.keys()
    assert not unknown, f"Unknown recorded scene roots: {unknown}"
    for asset in assets:
        keys = set(asset.get_scene_root_keys())
        selected = keys.intersection(pose_keys)
        required = not asset.is_anchor and (
            asset.get_spatial_relations() or (isinstance(asset, EmbodimentBase) and asset.get_relations())
        )
        if selected or required:
            missing = keys - pose_keys
            assert not missing, f"Recording is missing placed scene roots: {missing}"
        if selected:
            assert (
                get_relation(asset, RandomAroundSolution) is None
            ), f"Recorded object '{asset.name}' cannot randomize on reset"


def validate_root_reset_for_placement_replay(assets: list[PlaceableAsset]) -> None:
    """Require root-reset policies compatible with replacing them by recorded poses."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectBase

    for asset in assets:
        name = asset.get_scene_key()
        if isinstance(asset, Object):
            assert asset.reset_pose, f"Recorded asset '{name}' has pose resets disabled"
        if isinstance(asset, ObjectBase) and asset.initial_velocity is not None:
            velocity = asset.initial_velocity
            assert all(
                value == 0 for value in (*velocity.linear_xyz, *velocity.angular_xyz)
            ), f"Recorded asset '{name}' has nonzero initial velocity; replay resets velocity to zero"
        assert not asset.has_pose_reset_event() or isinstance(
            asset.get_initial_pose(), Pose
        ), f"Recorded asset '{name}' has a non-fixed pose-reset policy"


def _validate_native_samples(samples: list[PlacementSample]) -> None:
    """Require a nonempty sequence with consistent scene-root keys."""
    assert samples, "Placement samples must be nonempty"
    expected_keys = set(samples[0].poses)
    layout_ids: set[str] = set()
    for sample in samples:
        assert isinstance(sample, PlacementSample)
        assert set(sample.poses) == expected_keys, "Every placement sample must contain the same scene roots"
        assert sample.layout_id not in layout_ids, f"Duplicate placement layout_id: {sample.layout_id!r}"
        layout_ids.add(sample.layout_id)


def _validate_native_poses(poses: dict[str, Pose]) -> None:
    """Require finite, normalized poses keyed by nonempty scene-root names."""
    assert poses, "Placement sample must contain scene roots"
    assert all(isinstance(key, str) and key for key in poses), "Placement scene-root names must be nonempty"
    for pose in poses.values():
        assert isinstance(pose, Pose)
        components = (*pose.position_xyz, *pose.rotation_xyzw)
        assert all(
            isinstance(component, Real) and not isinstance(component, bool) and math.isfinite(component)
            for component in components
        ), "Placement poses must be finite"
        assert math.isclose(
            sum(component * component for component in pose.rotation_xyzw),
            1.0,
            abs_tol=1e-4,
        ), "Placement poses must have unit quaternions"
