# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import field, make_dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from isaaclab_arena.variations.sampler_base import SamplerBaseCfg
    from isaaclab_arena.variations.variation_base import VariationBase


# TODO(cvolk, 2026-07-08): Move variation composition under isaaclab_arena.hydra without changing behavior.

# Path, below a variation, of the field that selects the variation's sampler cfg type.
_SAMPLER_DISTRIBUTION_PATH = "sampler_cfg.distribution"


def overrides_from_dict(values: dict[str, Any]) -> list[str]:
    """Serialize nested variation values as dotted Hydra override strings."""
    overrides: list[str] = []
    stack: list[tuple[str, object]] = [("", values)]
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, dict):
            for key, child_value in value.items():
                assert key, "Hydra override keys must be non-empty"
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                stack.append((child_prefix, child_value))
            continue
        assert prefix, "Hydra override paths must be non-empty"
        overrides.append(f"{prefix}={json.dumps(value, separators=(',', ':'))}")
    return overrides


def split_variation_overrides(
    args: list[str], variations: dict[str, list[VariationBase]]
) -> tuple[list[str], list[str]]:
    """Separate registered host.variation overrides from other CLI arguments.

    Args:
        args: Unconsumed CLI arguments, including Hydra overrides.
        variations: Host names mapped to their attached variations.

    Returns:
        Variation overrides and remaining arguments, preserving order within each list.
    """
    variation_paths = set()
    for host, host_variations in variations.items():
        for variation in host_variations:
            variation_paths.add(f"{host}.{variation.name}")
    variation_overrides = []
    remaining_args = []
    for arg in args:
        key = arg.lstrip("+~").partition("=")[0]
        variation_path = ".".join(key.split(".")[:2])
        if variation_path in variation_paths and ("=" in arg or arg.startswith("~")):
            variation_overrides.append(arg)
        else:
            remaining_args.append(arg)
    return variation_overrides, remaining_args


def apply_overrides(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
) -> None:
    """Apply Hydra-style overrides to ``variations`` in-place.

    Builds a typed ``VariationsCfg`` with ``hydra_overrides`` and applies
    each per-variation cfg through ``VariationBase.apply_cfg``.

    Args:
        variations: ``{asset_name: [variation, ...]}`` mapping whose cfgs
            will be replaced by the values after applying overrides.
        hydra_overrides: Hydra override strings, dotted-path syntax
            mirroring the schema attribute paths. Example::

                apply_overrides(scene.get_asset_variations(), [
                    "cracker_box.color.enabled=true",
                    "cracker_box.color.sampler.low=[0.2,0.2,0.0]",
                    "cracker_box.color.sampler.high=[1.0,1.0,0.0]",
                ])
    """
    all_variations_cfg = build_variations_cfg_with_overrides(variations, hydra_overrides)
    if all_variations_cfg is None:
        return
    # Loop over the variations' cfgs and apply the overrides.
    for asset_name, asset_variations in variations.items():
        asset_cfg = getattr(all_variations_cfg, asset_name)
        for variation in asset_variations:
            variation_cfg = getattr(asset_cfg, variation.name)
            variation.apply_cfg(variation_cfg)


def build_variations_cfg_with_overrides(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
) -> Any | None:
    """Return typed variation configuration values after applying Hydra overrides.

    Builds the schema from copies of the current variation configs without updating the variations.

    Args:
        variations: ``{asset_name: [variation, ...]}`` the variations.
        hydra_overrides: Hydra override strings.

    Returns:
        The built ``VariationsCfg`` instance, including overridden values, or ``None`` when
        ``variations`` is empty.
    """
    selected_sampler_cfgs = _get_sampler_cfgs_selected_by_overrides(variations, hydra_overrides)
    variations_cfg_cls = _build_variations_cfg_type(variations, selected_sampler_cfgs)
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


def _get_sampler_cfgs_selected_by_overrides(
    variations: dict[str, list[VariationBase]],
    hydra_overrides: list[str],
) -> dict[str, SamplerBaseCfg]:
    """Return a default sampler cfg for each variation whose ``sampler_cfg.distribution`` override changes its sampler.

    Hydra overrides cannot change the type of a config node, so the schema must hold the selected sampler cfg type
    before the remaining overrides set its fields. When a variation has several such overrides, the last one wins.

    Args:
        variations: ``{asset_name: [variation, ...]}`` the variations.
        hydra_overrides: Hydra override strings.

    Returns:
        ``{"<asset_name>.<variation_name>": sampler_cfg}`` for each variation whose selected distribution differs from
        its current one.
    """
    distribution_overrides = [override for override in hydra_overrides if _SAMPLER_DISTRIBUTION_PATH in override]
    if not distribution_overrides:
        return {}
    distributions_by_variation_path: dict[str, Any] = {}
    for override in OverridesParser.create().parse_overrides(distribution_overrides):
        key_parts = override.key_or_group.split(".")
        if override.is_delete() or ".".join(key_parts[2:]) != _SAMPLER_DISTRIBUTION_PATH:
            continue
        distributions_by_variation_path[".".join(key_parts[:2])] = override.value()

    variations_by_path: dict[str, VariationBase] = {}
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            variations_by_path[f"{asset_name}.{variation.name}"] = variation
    selected_sampler_cfgs: dict[str, SamplerBaseCfg] = {}
    for variation_path, distribution in distributions_by_variation_path.items():
        variation = variations_by_path.get(variation_path)
        # Hydra composition reports an override of a variation that does not exist.
        if variation is None:
            continue
        # Selecting the current distribution keeps the current sampler cfg and its values.
        if getattr(variation.cfg.sampler_cfg, "distribution", None) == distribution:
            continue
        sampler_cfg_types = _get_sampler_cfg_types_by_distribution(variation)
        assert distribution in sampler_cfg_types, (
            f"'{variation_path}.{_SAMPLER_DISTRIBUTION_PATH}={distribution}' names no sampler that"
            f" {type(variation).__name__} accepts. Available distributions: {sorted(sampler_cfg_types)}."
        )
        selected_sampler_cfgs[variation_path] = sampler_cfg_types[distribution]()
    return selected_sampler_cfgs


def _get_sampler_cfg_types_by_distribution(variation: VariationBase) -> dict[str, type[SamplerBaseCfg]]:
    """Return the sampler cfg types that ``variation``'s ``sampler_cfg`` field accepts, keyed by distribution name."""
    # Deferred so that importing this module does not import the samplers.
    from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg  # noqa: PLC0415
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg  # noqa: PLC0415

    accepted_sampler_cfg_type = get_type_hints(type(variation.cfg))["sampler_cfg"]
    sampler_cfg_types: dict[str, type[SamplerBaseCfg]] = {}
    for sampler_cfg_type in (UniformSamplerCfg, ChoiceSamplerCfg):
        if issubclass(sampler_cfg_type, accepted_sampler_cfg_type):
            sampler_cfg_types[sampler_cfg_type().distribution] = sampler_cfg_type
    return sampler_cfg_types


def _build_variations_cfg_type(
    variations: dict[str, list[VariationBase]],
    selected_sampler_cfgs: dict[str, SamplerBaseCfg] | None = None,
) -> type | None:
    """Build a nested variation configuration dataclass type, or return ``None`` for an empty mapping.

    The class has one field per asset; each asset field's type is a
    dataclass whose fields are the attached variations' cfgs. Each
    per-variation field is typed as the variation's own Cfg class, i.e. a
    child class of ``*VariationBaseCfg``.

    Args:
        variations: ``{asset_name: [variation, ...]}`` dict.
        selected_sampler_cfgs: ``{"<asset_name>.<variation_name>": sampler_cfg}`` sampler cfgs that replace the
            variations' current ``sampler_cfg`` in the field defaults.

    Returns:
        The dynamically-built ``VariationsCfg`` dataclass type, or ``None``
        when ``variations`` is empty.
    """
    if not variations:
        return None
    selected_sampler_cfgs = selected_sampler_cfgs or {}

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
            selected_sampler_cfg = selected_sampler_cfgs.get(f"{asset_name}.{variation.name}")
            if selected_sampler_cfg is not None:
                default_cfg.sampler_cfg = selected_sampler_cfg
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
    sampler_hint = ""
    if any(".sampler_cfg." in override for override in hydra_overrides):
        sampler_hint = (
            "A sampler_cfg field must belong to the variation's sampler. Set '<host>.<variation>."
            f"{_SAMPLER_DISTRIBUTION_PATH}' (for example to 'choice') to change the sampler.\n"
        )
    raise ValueError(
        f"Unknown Hydra variation override ({override_hint}). "
        "No matching host or variation name in this environment.\n"
        f"{sampler_hint}"
        f"Available variation paths:\n{_format_available_variation_paths(variations)}\n"
        f"Original error:\n{cause}"
    ) from cause
