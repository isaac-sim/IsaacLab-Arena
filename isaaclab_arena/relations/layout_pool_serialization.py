# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JSON serialization for PooledObjectPlacer layout pools.

Owns the on-disk schema and its validation so the placer keeps only pool-state orchestration.
The pool is regenerable by re-solving, so a stale/incompatible file is meant to be re-saved
rather than migrated; load fails loudly on any structural problem instead of placing wrong poses.

On-disk schema (PoolDocument.to_dict / from_dict):
    {
      "placement_seed": int | null,        # restored on load so sampling reproduces the saved run
      "num_envs": int,
      "uses_env_specific_bboxes": bool,
      "had_fallbacks": bool,               # whether any stored layout was a best-loss fallback
      "env_pools": [                        # one list per env, outer length == num_envs
        [ {                                 # one entry per stored layout (serialize_layout)
            "positions": {obj_name: [x, y, z]},
            "orientations": {obj_name: yaw},
            "validation": {check_name: bool},
            "final_loss": float,
            "attempts": int,
        }, ... ],
      ],
    }
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_result import PlacementResult, ValidationReport

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase

_POOL_REQUIRED_KEYS = ("placement_seed", "num_envs", "uses_env_specific_bboxes", "had_fallbacks", "env_pools")
_LAYOUT_REQUIRED_KEYS = ("positions", "orientations", "validation", "final_loss", "attempts")


@dataclass(frozen=True)
class PoolDocument:
    """Pool-wide metadata plus raw per-env layout dicts.

    env_pools holds serialized per-layout dicts (see serialize_layout); the caller materializes
    them with deserialize_layout once it knows the live objects.
    """

    placement_seed: int | None
    num_envs: int
    uses_env_specific_bboxes: bool
    had_fallbacks: bool
    env_pools: list[list[dict]]

    def to_dict(self) -> dict:
        return {
            "placement_seed": self.placement_seed,
            "num_envs": self.num_envs,
            "uses_env_specific_bboxes": self.uses_env_specific_bboxes,
            "had_fallbacks": self.had_fallbacks,
            "env_pools": self.env_pools,
        }

    @classmethod
    def from_dict(cls, data: object, path: Path) -> PoolDocument:
        """Structurally validate a parsed document, naming the path on any problem."""
        assert isinstance(data, dict), f"Layout pool file is not a JSON object: {path}"
        for key in _POOL_REQUIRED_KEYS:
            assert key in data, f"Layout pool file is missing required key '{key}': {path}"

        env_pools = data["env_pools"]
        num_envs = data["num_envs"]
        seed = data["placement_seed"]
        assert isinstance(env_pools, list), f"Layout pool 'env_pools' must be a list: {path}"
        assert isinstance(num_envs, int) and not isinstance(
            num_envs, bool
        ), f"Layout pool 'num_envs' must be an int: {path}"
        assert seed is None or (
            isinstance(seed, int) and not isinstance(seed, bool)
        ), f"Layout pool 'placement_seed' must be int or null: {path}"
        assert isinstance(
            data["uses_env_specific_bboxes"], bool
        ), f"Layout pool 'uses_env_specific_bboxes' must be a bool: {path}"
        assert isinstance(data["had_fallbacks"], bool), f"Layout pool 'had_fallbacks' must be a bool: {path}"
        assert num_envs == len(env_pools), f"Corrupt layout pool: num_envs does not match env_pools length: {path}"
        for cur_env, env_layouts in enumerate(env_pools):
            assert isinstance(env_layouts, list), f"Layout pool env {cur_env} must be a list: {path}"
            for entry in env_layouts:
                assert isinstance(entry, dict), f"Layout pool env {cur_env} has a non-dict layout entry: {path}"
        return cls(
            placement_seed=seed,
            num_envs=num_envs,
            uses_env_specific_bboxes=data["uses_env_specific_bboxes"],
            had_fallbacks=data["had_fallbacks"],
            env_pools=env_pools,
        )


def write_pool_document(path: Path, document: PoolDocument) -> None:
    """Atomically write a pool document, failing loudly on non-finite values.

    Serializes to a string first (allow_nan=False) so a degenerate NaN/inf pose or loss raises
    before any file is touched, leaving no orphan temp file. Writes to a temp file then os.replace
    so an interrupted write never leaves a half-written file.
    """
    payload = json.dumps(document.to_dict(), indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(payload)
    os.replace(tmp_path, path)


def read_pool_document(path: Path) -> PoolDocument:
    """Read and structurally validate a pool document, naming the path on any problem.

    Validates only path-level structure; the caller still checks caller-dependent invariants
    (requested num_envs, heterogeneity, objects).
    """
    assert path.is_file(), f"Layout pool file not found: {path}"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Layout pool file is not valid JSON: {path}") from exc
    return PoolDocument.from_dict(data, path)


def serialize_layout(result: PlacementResult) -> dict:
    """Flatten one layout to JSON-safe primitives, keyed by object name."""
    return {
        "positions": {obj.name: list(pos) for obj, pos in result.positions.items()},
        "orientations": {obj.name: yaw for obj, yaw in result.orientations.items()},
        "validation": dict(result.validation.checks),
        "final_loss": result.final_loss,
        "attempts": result.attempts,
    }


def deserialize_layout(data: dict, name_to_obj: dict[str, ObjectBase]) -> PlacementResult:
    """Rebuild a PlacementResult, re-keying by the live objects that match each saved name."""
    for key in _LAYOUT_REQUIRED_KEYS:
        assert key in data, f"Serialized layout is missing required key '{key}'."

    positions_data = data["positions"]
    orientations_data = data["orientations"]
    checks = data["validation"]
    final_loss = data["final_loss"]
    attempts = data["attempts"]
    assert isinstance(
        positions_data, dict
    ), f"Serialized layout 'positions' must be a dict, got {type(positions_data).__name__}."
    assert isinstance(
        orientations_data, dict
    ), f"Serialized layout 'orientations' must be a dict, got {type(orientations_data).__name__}."
    assert isinstance(checks, dict), f"Serialized layout 'validation' must be a dict, got {type(checks).__name__}."
    assert checks, "Serialized layout has an empty validation map; it would load as a failing layout."
    for name, ok in checks.items():
        assert isinstance(ok, bool), f"Validation check '{name}' must be a JSON bool, got {type(ok).__name__}."
    assert (
        isinstance(final_loss, (int, float)) and not isinstance(final_loss, bool) and math.isfinite(final_loss)
    ), f"Serialized 'final_loss' must be a finite number, got {final_loss!r}."
    assert isinstance(attempts, int) and not isinstance(
        attempts, bool
    ), f"Serialized 'attempts' must be an int, got {attempts!r}."

    # Every live object must have a saved pose, and vice versa, so a stale file can't silently
    # leave an object at its origin (or reference one no longer in the scene).
    assert set(positions_data) == set(name_to_obj), (
        f"Saved layout objects {sorted(positions_data)} do not match the provided objects "
        f"{sorted(name_to_obj)}; re-solve instead of loading this cache."
    )
    assert set(orientations_data) <= set(
        positions_data
    ), f"Saved orientations {sorted(orientations_data)} are not a subset of positions {sorted(positions_data)}."

    def parse_position(name: str, pos: object) -> tuple[float, float, float]:
        assert (
            isinstance(pos, (list, tuple)) and len(pos) == 3
        ), f"Serialized position for '{name}' must be a length-3 sequence, got {pos!r}."
        assert all(
            isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in pos
        ), f"Serialized position for '{name}' must be finite numbers, got {pos!r}."
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def parse_yaw(name: str, yaw: object) -> float:
        assert (
            isinstance(yaw, (int, float)) and not isinstance(yaw, bool) and math.isfinite(yaw)
        ), f"Serialized orientation for '{name}' must be a finite number, got {yaw!r}."
        return float(yaw)

    positions = {name_to_obj[name]: parse_position(name, pos) for name, pos in positions_data.items()}
    orientations = {name_to_obj[name]: parse_yaw(name, yaw) for name, yaw in orientations_data.items()}
    return PlacementResult(
        positions=positions,
        orientations=orientations,
        validation=ValidationReport(checks=dict(checks)),
        final_loss=float(final_loss),
        attempts=attempts,
    )
