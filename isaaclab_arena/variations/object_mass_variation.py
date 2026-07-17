# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-env object mass variation.

Samples an absolute mass for a rigid object at reset and applies it through the
same mass/inertia setters used by Isaac Lab's rigid-body mass randomizer.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Smallest physically meaningful mass [kg]. Sampled masses below this would inject invalid
# (near-zero / NaN-prone) values into the physics engine, so they are rejected rather than clamped.
_MIN_PHYSICAL_MASS_KG = 1e-6


@configclass
class ObjectMassVariationCfg(VariationBaseCfg):
    """Configuration for ObjectMassVariation."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[0.05], high=[2.0]))
    """Uniform distribution over absolute object mass [kg]."""

    recompute_inertia: bool = True
    """Whether to scale inertia tensors by the sampled-mass/default-mass ratio."""


class ObjectMassVariation(RunTimeVariationBase):
    """Vary a rigid object's absolute mass at reset.

    Each reset samples one scalar mass per resetting environment. The event
    applies the sample directly, so the recorded variation value is the
    simulated mass.

    Args:
        asset_name: Scene-entity name of the target rigid object.
        cfg: Tunable parameters. Override the mass distribution via
            ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: ObjectMassVariationCfg

    def __init__(
        self,
        asset_name: str,
        cfg: ObjectMassVariationCfg | None = None,
        name: str = "mass",
    ):
        super().__init__(cfg=cfg if cfg is not None else ObjectMassVariationCfg(), name=name)
        self.asset_name = asset_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"ObjectMassVariation on '{self.asset_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        event_name = f"{self.asset_name}_mass_variation"
        event_cfg = EventTermCfg(
            func=ApplyObjectMassFromSampler,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "sampler": self._sampler,
                "recompute_inertia": self.cfg.recompute_inertia,
            },
        )
        return event_name, event_cfg


class ApplyObjectMassFromSampler(ManagerTermBase):
    """Event term: set a rigid object's absolute mass from sampler draws.

    The sampler must produce one scalar per environment. Defaults are snapshotted
    on the first call so repeated resets always apply the new mass relative to
    the original inertia tensor, not the previous reset's randomized tensor.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]

        self.asset = env.scene[self.asset_cfg.name]
        assert hasattr(self.asset, "set_masses_index") and hasattr(self.asset, "set_inertias_index"), (
            "ApplyObjectMassFromSampler expects a rigid object-like asset with mass/inertia setters at "
            f"scene['{self.asset_cfg.name}']; got {type(self.asset).__name__}."
        )
        assert tuple(sampler.shape_per_sample) == (1,), (
            "ApplyObjectMassFromSampler expects a sampler with shape_per_sample (1,) over absolute mass; "
            f"got {tuple(sampler.shape_per_sample)}."
        )
        selected_bodies_count = self._selected_body_count()
        assert selected_bodies_count == 1, (
            "ApplyObjectMassFromSampler currently supports exactly one selected body; "
            f"'{self.asset_cfg.name}' selected {selected_bodies_count} bodies."
        )

        self._default_mass: torch.Tensor | None = None
        self._default_inertia: torch.Tensor | None = None

    def _selected_body_count(self) -> int:
        """Return how many body IDs this term will update."""
        body_ids = self.asset_cfg.body_ids
        if body_ids == slice(None):
            return self.asset.num_bodies
        if isinstance(body_ids, int):
            return 1
        return len(body_ids)

    def _body_ids_tensor(self) -> torch.Tensor:
        """Return the selected body IDs as an index tensor."""
        body_ids = self.asset_cfg.body_ids
        if body_ids == slice(None):
            return torch.arange(self.asset.num_bodies, dtype=torch.int32, device=self.asset.device)
        if isinstance(body_ids, int):
            body_ids = [body_ids]
        return torch.tensor(body_ids, dtype=torch.int32, device=self.asset.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
        recompute_inertia: bool,
    ):
        if self._default_mass is None:
            self._default_mass = wp.to_torch(self.asset.data.body_mass).clone()
        if self._default_inertia is None:
            self._default_inertia = wp.to_torch(self.asset.data.body_inertia).clone()

        assert self._default_mass is not None
        assert self._default_inertia is not None

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.asset.device, dtype=torch.int32).reshape(-1)
        body_ids = self._body_ids_tensor()

        # TODO(tstuyck, 2026-07-17): The sampler draws on CPU, so this moves the samples to the sim
        # device every reset. Make ContinuousSampler device-aware (draw directly on device) to drop
        # this transfer here and in the other variations that copy sampler output onto the device.
        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        masses_to_apply = sample.to(device=self._default_mass.device, dtype=self._default_mass.dtype)
        if masses_to_apply.numel() > 0:
            assert torch.all(masses_to_apply >= _MIN_PHYSICAL_MASS_KG), (
                f"ObjectMassVariation sampled a mass below the minimum physical mass ({_MIN_PHYSICAL_MASS_KG} kg); "
                f"constrain sampler_cfg.low. min sampled mass={float(masses_to_apply.min())}."
            )

        # set_masses_index writes only the (env_ids, body_ids) subset, so pass the sampled masses
        # straight through instead of cloning the full body_mass tensor to slice the same subset back out.
        self.asset.set_masses_index(masses=masses_to_apply, body_ids=body_ids, env_ids=env_ids)

        if recompute_inertia:
            default_masses = self._default_mass[env_ids[:, None], body_ids]
            assert torch.all(default_masses > 0.0), (
                "ObjectMassVariation cannot recompute inertia from non-positive default mass values; "
                f"default masses={default_masses}."
            )
            # Scale each inertia tensor by the same factor the mass changed by (uniform-density
            # assumption: I ∝ m for a fixed shape), always relative to the snapshotted default inertia.
            ratios = masses_to_apply / default_masses
            new_inertias = self._default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            self.asset.set_inertias_index(inertias=new_inertias, body_ids=body_ids, env_ids=env_ids)
