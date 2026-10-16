# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that chooses which candidate asset a rigid Object spawns in a build."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import (
    BuildTimeVariationBase,
    VariationBaseCfg,
    get_enabled_variations_by_key,
)

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.variations.recorded_variation_samples import RebuildVariationRecord
    from isaaclab_arena.variations.variation_base import VariationBase


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Configuration for AssetSelectionVariation."""

    random_choice: bool = False
    """False takes name number ``seed % n`` of the ``n`` names in use, so consecutive build seeds take them in
    order. True draws one name at random, seeded by the build seed and the variation key."""

    asset_names: list[str] = field(default_factory=list)
    """Candidate names to choose from, in this order. Empty uses every candidate in declaration order."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Sampler that draws the candidate name."""


def assert_named_spawn_cfg(named_spawn_cfg: tuple[str, SpawnerCfg], owner: str) -> None:
    """Assert that ``named_spawn_cfg`` is a (name, spawn configuration) pair with one spawn configuration.

    Args:
        named_spawn_cfg: Pair to check, such as AssetRegistry.get_named_spawn_cfg() returns.
        owner: What received the pair, for example ``"Object 'pickup' asset="``. Starts each assert message.
    """
    assert isinstance(named_spawn_cfg, tuple) and len(named_spawn_cfg) == 2, (
        f"{owner}: pass a (name, spawn configuration) pair with one spawn configuration, such as"
        " AssetRegistry.get_named_spawn_cfg() returns."
    )
    asset_name, spawn_cfg = named_spawn_cfg
    assert (
        isinstance(asset_name, str) and asset_name
    ), f"{owner}: the asset name is {asset_name!r}; pass a non-empty string."
    assert isinstance(spawn_cfg, SpawnerCfg) and not isinstance(spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)), (
        f"{owner}: '{asset_name}' is a {type(spawn_cfg).__name__}; pass one spawn configuration, not a"
        " MultiAssetSpawnerCfg or MultiUsdFileCfg."
    )


class AssetSelectionVariation(BuildTimeVariationBase):
    """Give a rigid Object one of several candidate assets, the same in every environment of a build.

    The selected spawn configuration stays on the Object, so a later build of the same Object starts from it.
    Scene.export_to_usd and an ObjectReference into the Object read its declared usd_path, not the selected asset,
    so both fail for an Object declared without one.

    Args:
        asset_candidates: ``(name, spawn_cfg)`` pairs, for example from AssetRegistry.get_named_spawn_cfg.
            The spawn configurations are copied.
        cfg: Selection settings. Defaults to a disabled selection over every candidate, in order.
        name: Name of the variation on its Object, the second part of its key ``<object>.<name>``.
    """

    cfg: AssetSelectionVariationCfg

    def __init__(
        self,
        asset_candidates: list[tuple[str, SpawnerCfg]],
        cfg: AssetSelectionVariationCfg | None = None,
        name: str = "asset_selection",
    ):
        assert asset_candidates, "AssetSelectionVariation needs at least one (name, spawn_cfg) candidate."
        # Filled before super().__init__(), whose apply_cfg call checks asset_names against these names.
        self._spawn_cfgs_by_name: dict[str, SpawnerCfg] = {}
        for candidate_index, candidate in enumerate(asset_candidates):
            assert_named_spawn_cfg(candidate, owner=f"AssetSelectionVariation '{name}' candidate {candidate_index}")
            candidate_name, spawn_cfg = candidate
            assert candidate_name not in self._spawn_cfgs_by_name, f"Duplicate candidate name '{candidate_name}'."
            self._spawn_cfgs_by_name[candidate_name] = deepcopy(spawn_cfg)
        super().__init__(cfg=cfg if cfg is not None else AssetSelectionVariationCfg(), name=name)
        self._object: Object | None = None
        self._build_seed: int | None = None

    def apply_cfg(self, cfg: AssetSelectionVariationCfg) -> None:
        """Apply ``cfg`` after checking that ``asset_names`` holds declared candidate names, each once.

        Args:
            cfg: Selection settings to apply.
        """
        declared_names = list(self._spawn_cfgs_by_name)
        unknown_names = [asset_name for asset_name in cfg.asset_names if asset_name not in self._spawn_cfgs_by_name]
        assert not unknown_names, (
            f"AssetSelectionVariation '{self.name}' has unknown asset_names {unknown_names}; declared candidates:"
            f" {declared_names}."
        )
        assert len(set(cfg.asset_names)) == len(
            cfg.asset_names
        ), f"AssetSelectionVariation '{self.name}' repeats a name in asset_names {cfg.asset_names}."
        super().apply_cfg(cfg)

    def on_add_to_asset(self, asset: Asset) -> None:
        """Keep ``asset`` as the Object whose ``object_cfg.spawn`` this selection replaces when enabled.

        ``asset`` must be a rigid Object, not a RigidObjectSet, without another AssetSelectionVariation, and this
        selection must not be on another Object yet.

        Args:
            asset: Object to add the selection to.
        """
        # Imported here because object.py imports this module for assert_named_spawn_cfg, and the asset modules
        # load pxr.
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.assets.object_type import ObjectType

        assert not isinstance(asset, RigidObjectSet), (
            f"'{asset.name}' is a RigidObjectSet, which already spawns one member per environment."
            " Add AssetSelectionVariation to a rigid Object."
        )
        assert (
            isinstance(asset, Object) and asset.object_type == ObjectType.RIGID
        ), f"AssetSelectionVariation needs a rigid Object, but '{asset.name}' ({type(asset).__name__}) is not one."
        assert not any(
            isinstance(variation, AssetSelectionVariation) for variation in asset.get_variations()
        ), f"'{asset.name}' already has an AssetSelectionVariation; put all candidates in that one."
        assert (
            self._object is None
        ), f"AssetSelectionVariation '{self.name}' was already added to '{self._object.name}'; create one per Object."
        # Object.variations holds this selection too, and Isaac Lab's configclass validation follows attributes
        # without a cycle check, so a validated config must not hold this Object.
        self._object = asset

    def configure_at_build_time(self, seed: int, variation_key: str) -> None:
        # In-order selection takes name number seed % n, and VariationBase does not keep the seed.
        self._build_seed = seed
        super().configure_at_build_time(seed, variation_key)

    def _realize_at_build_time(self, generator: torch.Generator) -> None:
        assert (
            self._object is not None
        ), f"AssetSelectionVariation '{self.name}' was not added to an Object; add it with Object.add_variation()."
        # Checked at build time because bounding_box can be set after add_variation.
        assert self._object.bounding_box is None, (
            f"'{self._object.name}' sets a bounding_box override, which placement would use for every candidate."
            " Set bounding_box to None so placement reads the bounds of the selected asset."
        )
        names_in_use = list(self.cfg.asset_names) or list(self._spawn_cfgs_by_name)
        # Both modes call self.sampler.sample(), which reports the name to the variation recorder and returns the
        # recorded name during replay.
        if self.cfg.random_choice:
            selected_name = self.sampler.sample(1, names_in_use, generator=generator)[0]
        else:
            start_index = self._build_seed % len(names_in_use)
            selected_name = self.sampler.sample(1, names_in_use, start_index=start_index)[0]
        self._object.object_cfg.spawn = deepcopy(self._spawn_cfgs_by_name[selected_name])


def assert_selected_assets_replayed_with_placement(
    variations: dict[str, list[VariationBase]], variation_record: RebuildVariationRecord
) -> None:
    """Assert that a record with placement samples also holds a value for every enabled asset selection.

    Recorded poses were solved for the assets of the recording build, so a selection drawn live could
    put another asset at those poses.

    Args:
        variations: Variations of the environment, keyed by asset name.
        variation_record: Record loaded for recorded variation replay.
    """
    if not variation_record.has_placement_samples:
        return
    for variation_key, variation in get_enabled_variations_by_key(variations).items():
        if not isinstance(variation, AssetSelectionVariation):
            continue
        assert variation_key in variation_record.build_time_samples, (
            f"The replayed record has placement samples but no '{variation_key}' value. Replay a record"
            f" that holds both, or, if the poses were recorded with '{variation_key}' disabled, disable it here"
            " too."
        )
