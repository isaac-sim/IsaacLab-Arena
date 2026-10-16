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
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Configuration for AssetSelectionVariation."""

    random_choice: bool = False
    """True draws the candidate at random from the build seed; False takes the candidates in order, one step per
    build seed."""

    asset_names: list[str] = field(default_factory=list)
    """Candidate names to choose from, in this order. Empty uses every candidate in declaration order."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Sampler that draws the candidate name."""


class AssetSelectionVariation(BuildTimeVariationBase):
    """Give a rigid Object one of several candidate assets, the same in every environment of a build.

    Args:
        asset_candidates: ``(name, spawn_cfg)`` pairs, for example from AssetRegistry.get_asset_definition.
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
        for candidate in asset_candidates:
            assert isinstance(candidate, tuple) and len(candidate) == 2, f"Candidate {candidate!r} is not a pair."
            candidate_name, spawn_cfg = candidate
            assert isinstance(candidate_name, str) and candidate_name, "Candidate names must be non-empty strings."
            assert candidate_name not in self._spawn_cfgs_by_name, f"Duplicate candidate name '{candidate_name}'."
            assert isinstance(spawn_cfg, SpawnerCfg) and not isinstance(
                spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), f"Candidate '{candidate_name}' must have one spawn configuration."
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

    def attach_to_asset(self, asset: Asset) -> None:
        """Check that ``asset`` is a rigid Object that is not a RigidObjectSet, and keep it as the selection's Object.

        The Object must also have no bounding_box override and no other AssetSelectionVariation, and this
        selection must not be on another Object.

        Args:
            asset: Object to add the selection to.
        """
        # Imported here because the asset modules load pxr.
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
        assert (
            asset.bounding_box is None
        ), f"'{asset.name}' sets a bounding_box override, which placement would use instead of the candidate."
        assert not any(
            isinstance(variation, AssetSelectionVariation) for variation in asset.get_variations()
        ), f"'{asset.name}' already has an AssetSelectionVariation."
        assert (
            self._object is None
        ), f"AssetSelectionVariation '{self.name}' was already added to '{self._object.name}'; create one per Object."
        # Isaac Lab's configclass validation has no cycle check: a validated config holding this Object would recurse.
        self._object = asset

    def configure_at_build_time(self, seed: int, variation_key: str) -> None:
        """Select this build's candidate; in-order selection takes name number ``seed % n``, so keep ``seed``."""
        self._build_seed = seed
        super().configure_at_build_time(seed, variation_key)

    def _realize_at_build_time(self, generator: torch.Generator) -> None:
        assert self._object is not None, f"AssetSelectionVariation '{self.name}' was not added to an Object."
        asset_names = list(self.cfg.asset_names) or list(self._spawn_cfgs_by_name)
        # Both modes draw through the sampler because recording and replay hang off that call.
        if self.cfg.random_choice:
            selected_name = self.sampler.sample(1, asset_names, generator=generator)[0]
        else:
            start_index = self._build_seed % len(asset_names)
            selected_name = self.sampler.sample(1, asset_names, start_index=start_index)[0]
        self._object.object_cfg.spawn = deepcopy(self._spawn_cfgs_by_name[selected_name])
