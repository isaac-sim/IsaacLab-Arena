# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for turning Replicator segmentation metadata into tracking keys."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PATH_FIELD_NAMES = (
    "primPath",
    "prim_path",
    "path",
    "instancePath",
    "instance_path",
    "primPaths",
    "prim_paths",
    "instancePaths",
    "instance_paths",
)
_CLASS_FIELD_NAMES = ("class", "semanticLabel", "semantic_label", "label")


def get_label_for_instance_id(id_to_labels: Mapping[Any, Any], instance_id: int) -> Any | None:
    """Return a segmentation label, accepting either integer or string keys."""
    if instance_id in id_to_labels:
        return id_to_labels[instance_id]
    return id_to_labels.get(str(instance_id))


def label_to_tracking_candidates(label: Any) -> list[str]:
    """Return possible prim paths / semantic names from a Replicator label.

    Path-like fields are returned before semantic class labels because the class
    can be a coarse inherited tag such as ``"robot"`` while path fields often
    still point at the concrete mesh or link prim.
    """
    candidates: list[str] = []

    if isinstance(label, Mapping):
        for key in _PATH_FIELD_NAMES:
            _append_label_value(candidates, label.get(key))
        for key in _CLASS_FIELD_NAMES:
            _append_label_value(candidates, label.get(key))
    else:
        _append_label_value(candidates, label)

    return _deduplicate_nonempty(candidates)


def find_body_index_for_prim(prim_path: str, body_names: list[str]) -> int | None:
    """Find the articulation body whose name is most specifically present in a prim path."""
    path_parts = prim_path.strip("/").split("/")
    normalized_parts = [_normalize_token(part) for part in path_parts]
    best_idx: int | None = None
    best_score: tuple[int, int, int] | None = None

    for idx, body_name in enumerate(body_names):
        normalized_body = _normalize_token(body_name.rsplit("/", 1)[-1])
        if not normalized_body:
            continue
        for depth, normalized_part in enumerate(normalized_parts):
            match_rank = _body_match_rank(normalized_part, normalized_body)
            if match_rank is None:
                continue
            score = (depth, len(normalized_body), match_rank)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
    return best_idx


def _append_label_value(candidates: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        candidates.append(value)
        return
    if isinstance(value, Mapping):
        for key in (*_PATH_FIELD_NAMES, *_CLASS_FIELD_NAMES):
            _append_label_value(candidates, value.get(key))
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_label_value(candidates, item)
        return
    candidates.append(str(value))


def _deduplicate_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _body_match_rank(path_part: str, body_name: str) -> int | None:
    if path_part == body_name:
        return 2
    if path_part.startswith(f"{body_name}_") or path_part.endswith(f"_{body_name}"):
        return 1
    if f"_{body_name}_" in path_part:
        return 0
    return None
