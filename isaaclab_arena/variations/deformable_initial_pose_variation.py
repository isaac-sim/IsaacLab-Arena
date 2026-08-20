# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-reset deformable initial-pose offset variation."""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class DeformableInitialPoseVariationCfg(VariationBaseCfg):
    """Configuration for :class:`DeformableInitialPoseVariation`."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[-0.05, -0.05, 0.0], high=[0.05, 0.05, 0.0]),
    )
    """Uniform xyz offset [m] added to the deformable's default nodal state on reset."""


class DeformableInitialPoseVariation(RunTimeVariationBase):
    """Vary a deformable object's initial xyz offset at reset."""

    cfg: DeformableInitialPoseVariationCfg

    def __init__(
        self,
        asset_name: str,
        cfg: DeformableInitialPoseVariationCfg | None = None,
        name: str = "initial_pose",
    ):
        super().__init__(cfg=cfg if cfg is not None else DeformableInitialPoseVariationCfg(), name=name)
        self.asset_name = asset_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"DeformableInitialPoseVariation on '{self.asset_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        return (
            f"{self.asset_name}_initial_pose_variation",
            EventTermCfg(
                func=ApplyDeformableInitialPoseFromSampler,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.asset_name),
                    "sampler": self._sampler,
                },
            ),
        )


class ApplyDeformableInitialPoseFromSampler(ManagerTermBase):
    """Event term that applies sampled xyz offsets to a deformable's default nodal state."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]
        assert tuple(sampler.shape_per_sample) == (3,), (
            "DeformableInitialPoseVariation expects an xyz sampler with shape_per_sample (3,); "
            f"got {tuple(sampler.shape_per_sample)}."
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        sampler: ContinuousSampler,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.int64).reshape(-1)

        asset = env.scene[asset_cfg.name]
        offsets = sampler.sample(num_samples=len(env_ids), env_ids=env_ids).to(
            device=asset.device,
            dtype=asset.data.default_nodal_state_w.torch.dtype,
        )
        nodal_state = asset.data.default_nodal_state_w.torch[env_ids].clone()
        nodal_state[..., :3] += offsets[:, None, :]
        nodal_state[..., 3:] = 0.0
        asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
