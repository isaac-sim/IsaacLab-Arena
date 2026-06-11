# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
    # Compose all the config classes from all the variations, and apply the overrides.
    all_variations_cfg = compose_variations_cfg_and_apply_overrides(variations, hydra_overrides)
    if all_variations_cfg is None:
        return
    # Loop over the run-time variations and apply the overrides, in the cfg class.
    for asset_name, asset_variations in variations.items():
        asset_cfg = getattr(all_variations_cfg, asset_name)
        for variation in asset_variations:
            variation_cfg = getattr(asset_cfg, variation.name)
            variation.apply_cfg(variation_cfg)


def compose_variations_cfg_and_apply_overrides(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
) -> Any | None:
    """Compose the variation Hydra override strings into a typed ``VariationsCfg`` instance.

    Builds the schema and applies the passed overrides to it, returning the modified dataclass.

    Args:
        variations: ``{asset_name: [variation, ...]}`` the variations.
        hydra_overrides: Hydra override strings.

    Returns:
        The composed ``VariationsCfg`` instance, including overridden values, or ``None`` when
        ``variations`` is empty.
    """
    variations_cfg_cls = _compose_variation_cfgs(variations)
    if variations_cfg_cls is None:
        return None
    ConfigStore.instance().store(name="arena_variations_schema", node=variations_cfg_cls)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    try:
        with initialize(version_base=None, config_path=None):
            variations_cfg_dict = compose(config_name="arena_variations_schema", overrides=hydra_overrides)
    except ConfigCompositionException as exc:
        _raise_unknown_override_error(variations, hydra_overrides, exc)
    variations_cfg = OmegaConf.to_object(variations_cfg_dict)
    return variations_cfg


def _compose_variation_cfgs(variations: dict[str, list[VariationBase]]) -> type | None:
    """Return the dataclass describing variations configs, or ``None`` if the mapping is empty.

    The class has one field per asset; each asset field's type is a
    dataclass whose fields are the attached variations' cfgs. Each
    per-variation field is typed as the variation's own Cfg class, i.e. a
    child class of ``*VariationBaseCfg``.

    Args:
        variations: ``{asset_name: [variation, ...]}`` dict.

    Returns:
        The dynamically-built ``VariationsCfg`` dataclass type, or ``None``
        when ``variations`` is empty.
    """
    if not variations:
        return None

    # Loop over all the assets and their variations.
    asset_fields: list[tuple[str, type, Any]] = []
    for asset_name, asset_variations in variations.items():
        # TODO(alexmillane, 2026-06-11): Support asset names that are not valid Python identifiers.
        assert asset_name.isidentifier(), (
            f"Asset name '{asset_name}' must be a valid Python identifier to build its variations "
            "config schema; non-identifier asset names are not yet supported."
        )
        # The list of variation fields for this asset.
        variation_fields: list[tuple[str, type, Any]] = []
        # Make a field for each variation.
        for variation in asset_variations:
            cfg_cls = type(variation.cfg)
            default_cfg = deepcopy(variation.cfg)
            variation_fields.append((variation.name, cfg_cls, field(default_factory=lambda d=default_cfg: deepcopy(d))))
        # Make a dataclass for the asset.
        asset_cls_name = "".join(part.capitalize() for part in asset_name.split("_")) + "VariationsCfg"
        asset_cls = make_dataclass(asset_cls_name, variation_fields)
        # Add the asset dataclass to the list of fields for the combined dataclass
        asset_fields.append((asset_name, asset_cls, field(default_factory=asset_cls)))
    # Make a dataclass for all the variations.
    variations_cls = make_dataclass("VariationsCfg", asset_fields)
    return variations_cls


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
