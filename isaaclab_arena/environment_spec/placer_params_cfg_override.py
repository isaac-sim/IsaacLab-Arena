# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply serialized graph overrides to object-placer parameters."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from isaaclab_arena.hydra.config_override import apply_config_override

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams


def build_placer_params_from_override(override: dict[str, Any] | None) -> ObjectPlacerParams:
    """Build object-placer parameters from defaults and an optional serialized override."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    placer_params = ObjectPlacerParams(
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
    )
    if override is not None:
        placer_params = _apply_placer_params_override(placer_params, override)
    return placer_params


def _apply_placer_params_override(
    placer_params: ObjectPlacerParams,
    override: dict[str, Any],
) -> ObjectPlacerParams:
    """Return placer params with a data-only YAML override applied."""
    override = _drop_none_override_values(override)
    candidate = copy.deepcopy(placer_params)
    apply_config_override(
        candidate,
        override,
        override_name="placer_params",
        path="placer_params",
        allow_hydra_targets=False,
    )
    candidate.validate()
    return candidate


def _drop_none_override_values(value: Any) -> Any:
    """Drop null mapping entries so an explicit null use the dataclass default."""
    if not isinstance(value, dict):
        return value
    return {key: _drop_none_override_values(item) for key, item in value.items() if item is not None}
