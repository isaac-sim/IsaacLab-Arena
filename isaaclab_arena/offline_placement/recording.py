# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asset requirements for reusable placement recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass
class PlacementRecordingSummary:
    """Recording output and candidate acceptance counts."""

    output: Path | None
    """Written JSONL path, or None when too few candidates were accepted."""
    accepted: int
    """Number of candidates that passed all required checks."""
    attempted: int
    """Total sampled candidates, including solver failures."""
    rejections: dict[tuple[int, int], str]
    """Rejection reasons keyed by source (environment index, reset batch index)."""


def validate_recording_assets(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> None:
    """Require scene root ownership and reset policies compatible with reusable recordings.

    Args:
        env: Built environment whose writable physics roots will be recorded.
        assets: Placement and unplaced scene assets represented in the recording.
    """
    from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners
    from isaaclab_arena.relations.placement_layouts import validate_root_reset_for_cached_layouts
    from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

    env = env.unwrapped
    assert not has_heterogeneous_objects(assets), "Resolve object sets before recording reusable layouts"
    keys = set(env.scene.rigid_objects) | set(env.scene.articulations)
    owners = get_scene_root_owners(assets)
    assert keys <= owners.keys(), f"Pass scene_assets for unplaced scene roots: {keys - owners.keys()}"
    recorded_assets = []
    for asset in assets:
        owned_keys = set(asset.get_scene_root_keys())
        selected = owned_keys.intersection(keys)
        if selected or (not asset.is_anchor and asset.get_spatial_relations()):
            assert (
                owned_keys and owned_keys <= keys
            ), f"'{asset.name}' needs writable physics roots: {owned_keys - keys}"
        if selected:
            assert (
                get_relation(asset, RandomAroundSolution) is None
            ), f"'{asset.name}': remove RandomAroundSolution for cached replay"
            recorded_assets.append(asset)
    assert keys, "Recording requires rigid objects or articulations"
    validate_root_reset_for_cached_layouts(recorded_assets)
