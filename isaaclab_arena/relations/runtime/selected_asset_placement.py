# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reuse the placement solver with concrete, immutable geometry per selected asset combination."""

from __future__ import annotations

import copy
import hashlib
import torch
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
from isaaclab_arena.relations.runtime.placement_sampler import PlacementSampler
from isaaclab_arena.variations.sampler_base import SamplerBase

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.runtime.placement_sampler import PlacementSample
    from isaaclab_arena.variations.runtime_rigid_asset_selection_variation import RuntimeRigidAssetSelectionVariation


class SelectedAssetPlacementSampler(PlacementSampler):
    """Cache ordinary placement samplers by selected geometry within one build's placement context."""

    def __init__(
        self,
        assets: list[PlaceableAsset],
        scene_assets: Iterable[Asset],
        selections: dict[str, RuntimeRigidAssetSelectionVariation],
        num_envs: int,
        placer_params: ObjectPlacerParams,
    ) -> None:
        SamplerBase.__init__(self)
        self.assets = assets
        self.write_assets = [asset for asset in assets if not asset.is_anchor]
        self.selections = selections
        self.num_envs = num_envs
        self.last_results = {}
        self._scene_assets = list(scene_assets)
        self._params = copy.deepcopy(placer_params)
        self._samplers: dict[tuple, PlacementSampler] = {}
        self._construction_samples = {}
        # Freeze the complete placement context before construction poses are written to the live assets.
        sources = list(dict.fromkeys([*self._scene_assets, *self.assets]))
        self._templates = self._copy_assets(sources, {})
        self.placement_pool = None

    @property
    def last_fill_diagnostics(self):
        """Combine the latest fill of each encountered geometry's pool."""
        from isaaclab_arena.relations.placement_diagnostics import PlacementDiagnostics

        diagnostics = PlacementDiagnostics(num_envs=self.num_envs)
        for sampler in self._samplers.values():
            diagnostics.batches.extend(sampler.last_fill_diagnostics.batches)
        return diagnostics

    def _copy_assets(self, sources, geometry_by_role):
        copies = {asset: copy.copy(asset) for asset in sources}
        memo = {id(asset): clone for asset, clone in copies.items()}
        for original, clone in copies.items():
            if hasattr(original, "object_cfg"):
                clone.object_cfg = copy.deepcopy(original.object_cfg)
            if hasattr(original, "relations"):
                clone.relations = copy.deepcopy(original.relations, memo)
            if hasattr(original, "runtime_asset_selection"):
                clone.runtime_asset_selection = None
            if original.name in geometry_by_role:
                candidate = geometry_by_role[original.name]
                clone.object_cfg.spawn = copy.deepcopy(candidate.object_cfg.spawn)
                selection = self.selections[original.name]
                clone.bounding_box = selection.body_bounds_for_candidate(candidate)
                clone._spawn_cfg_geometry_by_key = {}
                clone._selected_placement_geometry = SpawnCfgGeometry(candidate.object_cfg.spawn)
        return copies

    def _key(self, geometry_by_role):
        return tuple(
            (name, selection.fingerprints[geometry_by_role[name]]) for name, selection in self.selections.items()
        )

    @torch.inference_mode(False)
    def _sampler_for(self, names_by_role):
        """Construct solver/IK buffers outside rollout inference mode so later validation can update them."""
        from isaaclab_arena.environments.relation_solver_interface import prepare_live_relation_placement

        key = self._key(names_by_role)
        if key not in self._samplers:
            # Copy the frozen originals and remap every relation target to its concrete role copy.
            geometry = {name: self.selections[name].candidates[selected] for name, selected in names_by_role.items()}
            copies = self._copy_assets(list(self._templates.values()), geometry)
            selected_assets = [copies[self._templates[asset]] for asset in self.assets]
            sampler = prepare_live_relation_placement(
                selected_assets,
                self.num_envs,
                copy.deepcopy(self._params),
                scene_assets=[copies[self._templates[asset]] for asset in self._scene_assets],
            )
            assert sampler is not None, "Runtime selection requires movable relation-placed assets"
            assert not sampler.placement_pool.had_fallbacks, f"No valid placement for selected assets {names_by_role}"
            self._samplers[key] = sampler
        sampler = self._samplers[key]
        self.placement_pool = sampler.placement_pool
        return sampler

    def prepare_live(self, num_envs: int, resolve_on_reset: bool) -> list[PlacementResult]:
        assert num_envs == self.num_envs
        self._params.resolve_on_reset = resolve_on_reset
        # Reserve the exact layouts that will be consumed on each environment's first reset.
        layouts = []
        for env_id in range(num_envs):
            names = self._names(env_id)
            sampler = self._sampler_for(names)
            sample = sampler.sample(1, torch.tensor([env_id]))[0]
            result = sampler.last_results[env_id]
            self._construction_samples[env_id] = (sample, result, self._key(names))
            layouts.append(self._map_result(result))
        return layouts

    def _names(self, env_id: int) -> dict[str, str]:
        return {name: selection.selected_names[env_id] for name, selection in self.selections.items()}

    def _map_result(self, result: PlacementResult) -> PlacementResult:
        by_name = {asset.name: asset for asset in self.assets}
        return replace(
            result,
            positions={by_name[asset.name]: value for asset, value in result.positions.items()},
            orientations={by_name[asset.name]: value for asset, value in result.orientations.items()},
        )

    def sample(
        self, num_samples: int, env_ids: torch.Tensor, *, geometry_by_env: dict[int, dict[str, str]] | None = None
    ) -> list[PlacementSample]:
        """Place absolute environment IDs using their selected concrete geometry.

        geometry_by_env optionally supplies registry names by scene role and absolute environment ID.
        Pool keys include all selected spawn fingerprints; each sampler retains the complete fixed placement context.
        """
        assert num_samples == len(env_ids)
        samples = []
        self.last_results = {}
        for env_id in env_ids.tolist():
            names = self._names(env_id) if geometry_by_env is None else geometry_by_env[env_id]
            sampler = self._sampler_for(names)
            reserved = self._construction_samples.pop(env_id, None)
            if reserved is not None and reserved[2] == self._key(names):
                sample, result, _ = reserved
            else:
                sample = sampler.sample(1, torch.tensor([env_id]))[0]
                result = sampler.last_results[env_id]
            assert result.success, f"No valid placement for env {env_id}, selected assets {names}"
            metadata = {
                name: {"registry_name": names[name], "fingerprint": selection.fingerprints[names[name]]}
                for name, selection in self.selections.items()
            }
            pool_id = hashlib.sha256(repr(self._key(names)).encode()).hexdigest()[:12]
            samples.append(replace(sample, layout_id=f"{pool_id}/{sample.layout_id}", selected_assets=metadata))
            self.last_results[env_id] = self._map_result(result)
        self._notify(samples, env_ids)
        return samples
