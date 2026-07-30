# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Human-readable printing of an environment's Hydra-configurable variations."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from isaaclab_arena.variations import variations_hydra
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase

# Shown when the whole environment has no variations.
_EMPTY_MESSAGE = "No variations attached to this environment.\n"
# Shown under an asset header when that asset has no variations.
_NO_ASSET_VARIATIONS = "  (no variations)"


def get_variations_catalogue_as_string(
    variations: dict[str, list[VariationBase]],
    *,
    hydra_overrides: list[str] | None = None,
) -> str:
    """Return a human-readable catalog of every variation in ``variations``.

    Returns a catalog that lists, for every variation, the exact Hydra override paths a user can copy onto the command line.
    Note that when ``hydra_overrides`` are supplied, the printed defaults
    reflect them (i.e. the catalog shows the *effective* post-override values).

    The helpers below build it up one asset at a time.

    Example output::

        Hydra-configurable variations
        ================================

        Asset: cracker_box
          color (ColorVariation, run-time)
            Enable: cracker_box.color.enabled=true  (default: False)
            Fields:
              cracker_box.color.sampler.low = [0.0,0.0,0.0]
              cracker_box.color.sampler.high = [1.0,1.0,1.0]

    Args:
        variations: ``{asset_name: [variation, ...]}`` the variations.
        hydra_overrides: Hydra override strings.

    Returns:
        Formatted catalog string.
    """
    if not variations:
        return _EMPTY_MESSAGE
    # Resolve the overrides into a single cfg up front, then read each field's
    # effective (post-override) default from it while formatting below.
    resolved_variations_cfg: Any | None = variations_hydra.compose_variations_cfg_and_apply_overrides(
        variations, hydra_overrides or []
    )
    lines = ["Variations (Hydra-configurable)", "=" * 32, ""]
    # Sort so the catalog is deterministic regardless of dict insertion order.
    for asset_name in sorted(variations.keys()):
        asset_variations: list[VariationBase] = variations[asset_name]
        asset_lines: list[str] = _asset_variations_as_string(
            asset_name,
            asset_variations,
            resolved_variations_cfg,
        )
        lines.extend(asset_lines)
    return "\n".join(lines).rstrip() + "\n"


def _asset_variations_as_string(
    asset_name: str,
    asset_variations: list[VariationBase],
    resolved_variations_cfg: Any | None,
) -> list[str]:
    """Return the catalog block (a list of text lines) for a single asset.

    One header line per asset, then a sub-block per attached variation (its
    enable flag plus any tunable fields).
    """
    lines = [f"Asset: {asset_name}"]
    if not asset_variations:
        lines.append(_NO_ASSET_VARIATIONS)
        lines.append("")
    else:
        for variation in asset_variations:
            # Every override path for this variation starts with this prefix.
            prefix = f"{asset_name}.{variation.name}"
            timing = _get_build_or_run_time_string(variation)
            lines.append(f"  {variation.name} ({type(variation).__name__}, {timing})")
            # The enable flag is always present, so it gets its own line above Fields.
            enabled_default = _get_field_default(resolved_variations_cfg, asset_name, variation.name, "enabled")
            lines.append(f"    Enable: {prefix}.enabled=true  (default: {enabled_default})")
            field_lines = _format_variation_fields(prefix, variation, resolved_variations_cfg, asset_name)
            if field_lines:
                lines.append("    Fields:")
                lines.extend(f"      {line}" for line in field_lines)
            lines.append("")  # Blank line separates variation sub-blocks.
    return lines


def _get_build_or_run_time_string(variation: VariationBase) -> str:
    """Return ``"build-time"`` / ``"run-time"`` describing when ``variation`` is applied."""
    if isinstance(variation, RunTimeVariationBase):
        return "run-time"
    if isinstance(variation, BuildTimeVariationBase):
        return "build-time"
    return "unknown"


def _get_field_default(
    resolved_variations_cfg: Any | None,
    asset_name: str,
    variation_name: str,
    field_name: str,
) -> Any:
    """Look up ``<asset>.<variation>.<field>`` in the resolved cfg.

    Walks the path one attribute at a time, returning ``None`` as soon as any
    segment is missing (or the cfg itself is ``None``) rather than raising.
    """
    if resolved_variations_cfg is None:
        return None
    asset_cfg = getattr(resolved_variations_cfg, asset_name, None)
    if asset_cfg is None:
        return None
    variation_cfg = getattr(asset_cfg, variation_name, None)
    if variation_cfg is None:
        return None
    return getattr(variation_cfg, field_name, None)


def _format_variation_fields(
    prefix: str,
    variation: VariationBase,
    resolved_variations_cfg: Any | None,
    asset_name: str,
) -> list[str]:
    """Return the tunable cfg override paths for ``variation`` (excluding ``enabled``).

    Prefers the resolved cfg (so printed values reflect any overrides); falls
    back to the variation's own live cfg when the resolved cfg is unavailable.
    Either source is converted to a plain dict and flattened into dotted paths.
    """
    # Preferred source: the variation's slice of the resolved (post-override) cfg.
    if resolved_variations_cfg is not None:
        asset_cfg = getattr(resolved_variations_cfg, asset_name, None)
        variation_cfg = getattr(asset_cfg, variation.name, None) if asset_cfg is not None else None
        if variation_cfg is not None:
            data = _cfg_to_plain(variation_cfg)
            return _flatten_paths(prefix, data, skip_keys={"enabled"})
    # Fallback: the variation's own live cfg (no overrides resolved).
    data = _cfg_to_plain(variation.cfg)
    return _flatten_paths(prefix, data, skip_keys={"enabled"})


def _flatten_paths(prefix: str, data: dict[str, Any], *, skip_keys: set[str]) -> list[str]:
    """Flatten a nested cfg dict into sorted ``dotted.path = value`` lines.

    ``skip_keys`` are dropped at the current level only (nested levels keep all
    keys), which is how the top-level ``enabled`` flag is excluded here while
    still appearing under its own "Enable:" line elsewhere.
    """
    lines: list[str] = []
    for key in sorted(data.keys()):
        if key in skip_keys:
            continue
        path = f"{prefix}.{key}"
        value = data[key]
        # Recurse into non-empty sub-dicts; leaves become a single path = value line.
        if isinstance(value, dict) and value:
            lines.extend(_flatten_paths(path, value, skip_keys=set()))
        else:
            lines.append(f"{path} = {_format_value(value)}")
    return lines


def _format_value(value: Any) -> str:
    """Format a cfg value as a Hydra-CLI-compatible string.

    Lists are rendered without spaces after commas so the output can be pasted
    directly onto the command line (e.g. ``[-0.005,-0.005,-0.005]``).
    """
    if isinstance(value, list):
        inner = ",".join(_format_value(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, float):
        # Round to at most 3 significant figures, then re-parse so the repr stays
        # in plain (non-scientific) form for the ranges seen here (e.g. 3.14, 2000.0).
        return repr(float(f"{value:.3g}"))
    return repr(value)


def _cfg_to_plain(obj: Any) -> Any:
    """Recursively convert configclass / dataclass instances to plain Python types.

    Normalises the many cfg representations (configclass, dataclass, list, dict)
    into nested dicts/lists/scalars so :func:`_flatten_paths` has one shape to walk.
    """
    if hasattr(obj, "to_dict"):  # Isaac Lab configclass.
        return obj.to_dict()
    if is_dataclass(obj) and not isinstance(obj, type):  # Plain dataclass instance (not the class itself).
        return {f.name: _cfg_to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_cfg_to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _cfg_to_plain(value) for key, value in obj.items()}
    return obj  # Scalar / already-plain value.
