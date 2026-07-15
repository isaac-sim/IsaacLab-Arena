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


@configclass
class ObjectMassVariationCfg(VariationBaseCfg):
    """Configuration for ObjectMassVariation."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[0.05], high=[2.0]))
    """Uniform distribution over absolute object mass [kg]."""

    recompute_inertia: bool = True
    """Whether to scale inertia tensors by the sampled-mass/default-mass ratio."""

    min_mass: float = 1e-6
    """Minimum allowed sampled mass [kg]. Values below this fail instead of being clamped."""


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

    def apply_cfg(self, cfg: ObjectMassVariationCfg) -> None:
        """Apply and validate the mass variation config."""
        super().apply_cfg(cfg)
        assert (
            cfg.min_mass >= 1e-6
        ), f"ObjectMassVariation requires min_mass >= 1e-6 to avoid invalid physics masses; got {cfg.min_mass}."

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"ObjectMassVariation on '{self.asset_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        event_name = f"{self.asset_name}_mass_variation"
        event_cfg = EventTermCfg(
            func=apply_object_mass_from_sampler,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "sampler": self._sampler,
                "recompute_inertia": self.cfg.recompute_inertia,
                "min_mass": self.cfg.min_mass,
            },
        )
        return event_name, event_cfg


class apply_object_mass_from_sampler(ManagerTermBase):
    """Event term: set a rigid object's absolute mass from sampler draws.

    The sampler must produce one scalar per environment. Defaults are snapshotted
    on the first call so repeated resets always apply the new mass relative to
    the original inertia tensor, not the previous reset's randomized tensor.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]
        min_mass: float = cfg.params["min_mass"]

        self.asset = env.scene[self.asset_cfg.name]
        assert hasattr(self.asset, "set_masses_index") and hasattr(self.asset, "set_inertias_index"), (
            "apply_object_mass_from_sampler expects a rigid object-like asset with mass/inertia setters at "
            f"scene['{self.asset_cfg.name}']; got {type(self.asset).__name__}."
        )
        assert tuple(sampler.shape_per_sample) == (1,), (
            "apply_object_mass_from_sampler expects a sampler with shape_per_sample (1,) over absolute mass; "
            f"got {tuple(sampler.shape_per_sample)}."
        )
        assert (
            min_mass >= 1e-6
        ), f"apply_object_mass_from_sampler requires min_mass >= 1e-6 to avoid invalid physics masses; got {min_mass}."
        selected_bodies = self._selected_body_count()
        assert selected_bodies == 1, (
            "apply_object_mass_from_sampler currently supports exactly one selected body; "
            f"'{self.asset_cfg.name}' selected {selected_bodies} bodies."
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
        min_mass: float,
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

        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        masses_to_apply = sample.to(device=self._default_mass.device, dtype=self._default_mass.dtype)
        if masses_to_apply.numel() > 0:
            assert torch.all(masses_to_apply >= min_mass), (
                "ObjectMassVariation sampled a mass below min_mass. Constrain sampler_cfg.low or lower min_mass "
                f"within the supported range. min sampled mass={float(masses_to_apply.min())}, min_mass={min_mass}."
            )

        masses = wp.to_torch(self.asset.data.body_mass).clone()
        masses[env_ids[:, None], body_ids] = masses_to_apply
        self.asset.set_masses_index(masses=masses[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)

        if recompute_inertia:
            default_masses = self._default_mass[env_ids[:, None], body_ids]
            assert torch.all(default_masses > 0.0), (
                "ObjectMassVariation cannot recompute inertia from non-positive default mass values; "
                f"default masses={default_masses}."
            )
            ratios = masses_to_apply / default_masses
            inertias = wp.to_torch(self.asset.data.body_inertia).clone()
            inertias[env_ids[:, None], body_ids] = self._default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            self.asset.set_inertias_index(
                inertias=inertias[env_ids[:, None], body_ids],
                body_ids=body_ids,
                env_ids=env_ids,
            )
