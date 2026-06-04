# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Human-readable catalog of Hydra-configurable variations."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from isaaclab_arena.variations import variations_hydra
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase

_EMPTY_MESSAGE = "No variations attached to this environment.\n"
_NO_HOST_VARIATIONS = "  (no Hydra-configurable variations)"


def get_variations_catalogue_as_string(
    variations: dict[str, list[VariationBase]],
    *,
    hydra_overrides: list[str] | None = None,
) -> str:
    """Return a human-readable catalog of every variation in ``variations``.

    Args:
        variations: ``{host_name: [variation, ...]}`` mapping.
        hydra_overrides: Optional Hydra tokens; when set, composed defaults reflect them.

    Returns:
        Formatted catalog string.
    """
    if not variations:
        return _EMPTY_MESSAGE
    composed = variations_hydra.load_cfg_from_flags(variations, hydra_overrides or [])
    lines = ["Hydra-configurable variations", "=" * 32, ""]
    for host_name in sorted(variations.keys()):
        host_variations = variations[host_name]
        lines.append(f"Host: {host_name}")
        if not host_variations:
            lines.append(_NO_HOST_VARIATIONS)
            lines.append("")
            continue
        for variation in host_variations:
            prefix = f"{host_name}.{variation.name}"
            timing = _variation_timing(variation)
            lines.append(f"  {variation.name} ({type(variation).__name__}, {timing})")
            enabled_default = _get_field_default(composed, host_name, variation.name, "enabled")
            lines.append(f"    Enable: {prefix}.enabled=true  (default: {enabled_default})")
            field_lines = _format_variation_fields(prefix, variation, composed, host_name)
            if field_lines:
                lines.append("    Fields:")
                lines.extend(f"      {line}" for line in field_lines)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _variation_timing(variation: VariationBase) -> str:
    if isinstance(variation, RunTimeVariationBase):
        return "run-time"
    if isinstance(variation, BuildTimeVariationBase):
        return "build-time"
    return "unknown"


def _get_field_default(composed: Any | None, host_name: str, variation_name: str, field_name: str) -> Any:
    if composed is None:
        return None
    asset_cfg = getattr(composed, host_name, None)
    if asset_cfg is None:
        return None
    variation_cfg = getattr(asset_cfg, variation_name, None)
    if variation_cfg is None:
        return None
    return getattr(variation_cfg, field_name, None)


def _format_variation_fields(
    prefix: str,
    variation: VariationBase,
    composed: Any | None,
    host_name: str,
) -> list[str]:
    """Emit tunable cfg paths excluding the top-level ``enabled`` flag."""
    if composed is not None:
        asset_cfg = getattr(composed, host_name, None)
        variation_cfg = getattr(asset_cfg, variation.name, None) if asset_cfg is not None else None
        if variation_cfg is not None:
            data = _cfg_to_plain(variation_cfg)
            return _flatten_paths(prefix, data, skip_keys={"enabled"})
    data = _cfg_to_plain(variation.cfg)
    return _flatten_paths(prefix, data, skip_keys={"enabled"})


def _flatten_paths(prefix: str, data: dict[str, Any], *, skip_keys: set[str]) -> list[str]:
    lines: list[str] = []
    for key in sorted(data.keys()):
        if key in skip_keys:
            continue
        path = f"{prefix}.{key}"
        value = data[key]
        if isinstance(value, dict) and value:
            lines.extend(_flatten_paths(path, value, skip_keys=set()))
        else:
            lines.append(f"{path} = {value!r}")
    return lines


def _cfg_to_plain(obj: Any) -> Any:
    """Recursively convert configclass / dataclass instances to plain Python types."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _cfg_to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_cfg_to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _cfg_to_plain(value) for key, value in obj.items()}
    return obj
