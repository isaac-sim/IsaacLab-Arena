# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from isaaclab_arena.hydra.config_override import apply_config_override, dotlist_to_override, nested_override

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase


def apply_overrides(
    variations: dict[str, list[VariationBase]],
    overrides: dict[str, Any] | list[str],
) -> None:
    """Apply nested or Hydra dotlist overrides to ``variations`` in place.

    Args:
        variations: ``{asset_name: [variation, ...]}`` mapping whose cfgs
            will be replaced by the composed values.
        overrides: A nested mapping or Hydra dotlist strings mirroring the
            variation paths. Example::

                apply_overrides(scene.get_asset_variations(), [
                    "cracker_box.color.enabled=true",
                    "cracker_box.color.sampler_cfg.low=[0.2,0.2,0.0]",
                    "cracker_box.color.sampler_cfg.high=[1.0,1.0,0.0]",
                ])
    """
    resolved_cfgs = resolve_overrides(variations, overrides)
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            variation.apply_cfg(resolved_cfgs[asset_name][variation.name])


def resolve_overrides(
    variations: dict[str, list[VariationBase]],
    overrides: dict[str, Any] | list[str],
) -> dict[str, dict[str, Any]]:
    """Return copied variation configurations with overrides applied.

    Args:
        variations: ``{asset_name: [variation, ...]}`` the variations.
        overrides: Nested override mapping or Hydra dotlist strings.

    Returns:
        Configurations keyed by host and variation name.
    """
    for asset_name in variations:
        assert asset_name.isidentifier(), (
            f"Asset name '{asset_name}' must be a valid Python identifier to build its variations config schema; "
            "non-identifier asset names are not yet supported."
        )

    override_values = dotlist_to_override(overrides) if isinstance(overrides, list) else nested_override(overrides)
    resolved_cfgs = {
        asset_name: {variation.name: deepcopy(variation.cfg) for variation in asset_variations}
        for asset_name, asset_variations in variations.items()
    }
    try:
        apply_config_override(resolved_cfgs, override_values)
    except (AssertionError, TypeError, ValueError) as exc:
        _raise_unknown_override_error(variations, overrides, exc)
    return resolved_cfgs


def _format_available_variation_paths(variations: dict[str, list[VariationBase]]) -> str:
    lines: list[str] = []
    for host_name in sorted(variations.keys()):
        for variation in variations[host_name]:
            lines.append(f"  {host_name}.{variation.name}")
    return "\n".join(lines) if lines else "  (none)"


def _raise_unknown_override_error(
    variations: dict[str, list[VariationBase]],
    overrides: dict[str, Any] | list[str],
    cause: Exception,
) -> None:
    override_hint = ", ".join(overrides) if isinstance(overrides, list) else repr(overrides)
    raise ValueError(
        f"Unknown Hydra variation override ({override_hint}). "
        "No matching host or variation name in this environment.\n"
        f"Available variation paths:\n{_format_available_variation_paths(variations)}\n"
        f"Original error:\n{cause}"
    ) from cause
