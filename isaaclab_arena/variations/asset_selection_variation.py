# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Choose fixed asset identities for a named scene object at build time."""

from __future__ import annotations

import hashlib
import torch
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.sim import MultiAssetSpawnerCfg, MultiUsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.sampler_base import SamplerBaseCfg
from isaaclab_arena.variations.sequential_choice_sampler import SequentialChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg, VariationBuildContext

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object


@configclass
class AssetSelectionVariationCfg(VariationBaseCfg):
    """Choose assets separately per environment, or share one choice across the build."""

    sample_per_environment: bool = True
    """Whether each environment receives its own fixed choice."""

    sampler_cfg: SamplerBaseCfg = field(default_factory=SequentialChoiceSamplerCfg)
    """Sequential assignment by default; ChoiceSamplerCfg enables independent random choices."""


class AssetSelectionVariation(BuildTimeVariationBase):
    """Copy candidate names and spawn settings, then sample names once per build.

    Attach to an ordinary rigid Object with add_variation(). Candidate instances
    provide definitions only; their poses, relations, and variations are not copied.

    Args:
        candidates: Ordered concrete rigid objects with unique, nonempty names.
        cfg: Optional sampling configuration; selection starts disabled.
        name: Variation key on the target object.
    """

    supported_sample_per_environment = (False, True)
    cfg: AssetSelectionVariationCfg

    def __init__(
        self,
        candidates: list[Object],
        cfg: AssetSelectionVariationCfg | None = None,
        name: str = "asset_selection",
    ) -> None:
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.assets.object_type import ObjectType

        assert candidates, "Asset selection requires at least one candidate."
        candidate_names = []
        candidate_spawn_configs = []
        for candidate in candidates:
            assert isinstance(candidate, Object) and not isinstance(
                candidate, RigidObjectSet
            ), "Asset selection candidates must be concrete rigid Object instances."
            assert candidate.object_type == ObjectType.RIGID, "Asset selection supports rigid candidates only."
            assert isinstance(candidate.name, str) and candidate.name.strip(), "Candidate names must be nonempty."
            assert (
                candidate.name not in candidate_names
            ), f"Duplicate asset selection candidate name: {candidate.name!r}."
            assert not isinstance(
                candidate.spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
            ), "Asset selection candidates must have one concrete native spawn configuration."
            assert candidate.bounding_box is None, "Asset selection candidates cannot have custom bounding boxes."
            candidate._assert_asset_selection_resolved()
            candidate_names.append(candidate.name)
            candidate_spawn_configs.append(deepcopy(candidate.spawn_cfg))
        self._candidate_names = tuple(candidate_names)
        self._candidate_spawn_configs = tuple(candidate_spawn_configs)
        self._selected_candidate_names: tuple[str, ...] | None = None
        super().__init__(cfg=cfg if cfg is not None else AssetSelectionVariationCfg(), name=name)

    @property
    def candidate_names(self) -> tuple[str, ...]:
        """Candidate recording IDs, copied in declaration order."""
        return self._candidate_names

    @property
    def selected_candidate_names(self) -> tuple[str, ...] | None:
        """Names sampled for this build; a shared selection contains one name."""
        return self._selected_candidate_names

    def get_candidate_spawn_configs(self) -> list[SpawnerCfg]:
        """Return independent native candidate configurations for build resolution."""
        return [deepcopy(spawn_cfg) for spawn_cfg in self._candidate_spawn_configs]

    def apply_cfg(self, cfg: AssetSelectionVariationCfg) -> None:
        assert self._selected_candidate_names is None, "Asset selection is already sampled; create a fresh environment."
        assert isinstance(
            cfg.sampler_cfg, (SequentialChoiceSamplerCfg, ChoiceSamplerCfg)
        ), "Asset selection supports SequentialChoiceSamplerCfg or ChoiceSamplerCfg."
        super().apply_cfg(cfg)

    def _validate_attachment(self, asset: Asset) -> None:
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.assets.object_type import ObjectType

        assert isinstance(asset, Object) and not isinstance(
            asset, RigidObjectSet
        ), "Attach asset selection to an ordinary rigid Object."
        assert asset.object_type == ObjectType.RIGID, "Asset selection supports rigid targets only."
        assert not isinstance(
            asset.spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
        ), "Asset selection targets must start with one concrete native spawn configuration."
        assert asset.bounding_box is None, "Asset selection targets cannot have custom bounding boxes."
        assert all(
            variation is self or not isinstance(variation, AssetSelectionVariation)
            for variation in asset.get_variations()
        ), f"Object '{asset.name}' already has an asset selection variation."

    def _realize_at_build_time(self, context: VariationBuildContext | None = None) -> None:
        assert context is not None, "Asset selection requires a VariationBuildContext."
        assert self.attached_asset is not None, "Attach asset selection to an Object before building."
        assert self._selected_candidate_names is None, "Asset selection is already sampled; create a fresh environment."
        assert self.enabled, "Enable asset selection before sampling."
        num_samples = context.num_envs if self.cfg.sample_per_environment else 1
        env_ids = torch.arange(context.num_envs) if self.cfg.sample_per_environment else None
        generator = torch.Generator(device="cpu")
        if context.seed is None:
            generator.seed()
        else:
            seed_bytes = hashlib.sha256(f"{context.seed}:{context.variation_key}".encode()).digest()
            generator.manual_seed(int.from_bytes(seed_bytes[:8], "big"))
        self._selected_candidate_names = tuple(
            self.sampler.sample(num_samples, self._candidate_names, env_ids=env_ids, generator=generator)
        )
