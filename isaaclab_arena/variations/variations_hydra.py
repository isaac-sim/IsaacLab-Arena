# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hydra-overridable cfg composition for a scene's variations.

These helpers build a typed Hydra schema from a ``{asset_name: [variation, ...]}``
mapping, compose override strings against that schema, and push the
resulting per-variation cfgs back through
:meth:`~isaaclab_arena.variations.variation_base.VariationBase.apply_cfg`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field, make_dataclass
from typing import TYPE_CHECKING, Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase


def _asset_class_name(asset_name: str) -> str:
    """Convert ``"cracker_box"`` to ``"CrackerBoxVariationsCfg"``."""
    camel = "".join(part.capitalize() for part in asset_name.split("_"))
    return f"{camel}VariationsCfg"


def build_schema(variations: dict[str, list[VariationBase]]) -> type | None:
    """Return the dataclass describing ``variations``, or ``None`` if the mapping is empty.

    The class has one field per asset; each asset field's type is a
    dataclass whose fields are the attached variations' cfgs. Each
    per-variation field is typed as the variation's own ``*Cfg`` and
    pre-populated by deep-copying its current live cfg, so override paths
    line up one-to-one with cfg attribute paths.

    Args:
        variations: ``{asset_name: [variation, ...]}`` mapping, typically
            from :meth:`~isaaclab_arena.scene.scene.Scene.get_asset_variations`.

    Returns:
        The dynamically-built ``VariationsCfg`` dataclass type, or ``None``
        when ``variations`` is empty.
    """
    if not variations:
        return None

    asset_fields: list[tuple[str, type, Any]] = []
    for asset_name, asset_variations in variations.items():
        variation_fields: list[tuple[str, type, Any]] = []
        for variation in asset_variations:
            cfg_cls = type(variation.cfg)
            default_cfg = deepcopy(variation.cfg)
            variation_fields.append((variation.name, cfg_cls, field(default_factory=lambda d=default_cfg: deepcopy(d))))
        asset_cls = make_dataclass(_asset_class_name(asset_name), variation_fields)
        asset_fields.append((asset_name, asset_cls, field(default_factory=asset_cls)))
    return make_dataclass("VariationsCfg", asset_fields)


def load_cfg_from_flags(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
) -> Any | None:
    """Compose Hydra override strings into a typed ``VariationsCfg`` instance.

    Builds the schema and applies the passed overrides to it, returning the modified dataclass.

    Args:
        variations: ``{asset_name: [variation, ...]}`` mapping that defines
            the schema shape.
        hydra_overrides: Hydra override strings. See :func:`apply_overrides`
            for examples.

    Returns:
        The composed ``VariationsCfg`` instance, or ``None`` when
        ``variations`` is empty.
    """
    schema_cls = build_schema(variations)
    if schema_cls is None:
        return None
    ConfigStore.instance().store(name="arena_variations_schema", node=schema_cls)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    try:
        with initialize(version_base=None, config_path=None):
            composed = compose(config_name="arena_variations_schema", overrides=hydra_overrides)
    except ConfigCompositionException as exc:
        _raise_unknown_override_error(variations, hydra_overrides, exc)
    return OmegaConf.to_object(composed)


def _format_available_variation_paths(variations: dict[str, list[VariationBase]]) -> str:
    lines: list[str] = []
    for host_name in sorted(variations.keys()):
        for variation in variations[host_name]:
            lines.append(f"  {host_name}.{variation.name}")
    return "\n".join(lines) if lines else "  (none)"


def _raise_unknown_override_error(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
    cause: ConfigCompositionException,
) -> None:
    override_hint = ", ".join(hydra_overrides)
    raise ValueError(
        f"Unknown Hydra variation override ({override_hint}). "
        "No matching host or variation name in this environment.\n"
        f"Available variation paths:\n{_format_available_variation_paths(variations)}"
    ) from cause


def apply_overrides(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
) -> None:
    """Apply Hydra-style overrides to ``variations`` in-place.

    Composes ``hydra_overrides`` into a typed ``VariationsCfg`` and pushes
    each per-variation cfg through
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.apply_cfg`.

    Args:
        variations: ``{asset_name: [variation, ...]}`` mapping whose cfgs
            will be replaced by the composed values.
        hydra_overrides: Hydra override strings, dotted-path syntax
            mirroring the schema attribute paths. Example::

                apply_overrides(scene.get_asset_variations(), [
                    "cracker_box.color.enabled=true",
                    "cracker_box.color.sampler.low=[0.2,0.2,0.0]",
                    "cracker_box.color.sampler.high=[1.0,1.0,0.0]",
                ])
    """
    composed = load_cfg_from_flags(variations, hydra_overrides)
    if composed is None:
        return
    for asset_name, asset_variations in variations.items():
        asset_cfg = getattr(composed, asset_name)
        for variation in asset_variations:
            variation_cfg = getattr(asset_cfg, variation.name)
            variation.apply_cfg(variation_cfg)
