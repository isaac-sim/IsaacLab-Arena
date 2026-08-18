# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-env object yaw variation.

Rotates a rigid object about world Z at reset, on top of whatever pose the placement solver wrote.
Object yaw is the first-order factor for a parallel-jaw grasp — the closable span across a box
footprint ``a x b`` at relative yaw ``phi`` is ``a|cos phi| + b|sin phi|`` — yet Arena exposes no
way to vary it: ``ObjectPlacerParams.random_yaw_init`` defaults to False and is not reachable from
``ArenaEnvBuilderCfg``. Every episode of every published run therefore shares one object yaw.

Being a ``Variation`` rather than a placer flag is the point: sampled values are recorded per episode
into the ``variations`` block, so yaw becomes a factor the sensitivity analysis can condition on.
"""

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


def rotate_quat_xyzw_about_z(quat_xyzw: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Pre-multiply ``quat_xyzw`` by a rotation of ``yaw`` radians about world Z.

    Written out rather than routed through ``isaaclab.utils.math``, whose helpers use the ``(w,x,y,z)``
    convention while ``root_link_pose_w`` is ``(x,y,z,w)``. Converting twice to borrow a one-line
    helper is a good way to introduce a silent transposition; see ``test_object_yaw_variation``.
    """
    x, y, z, w = quat_xyzw.unbind(-1)
    c, s = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        [c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z],
        dim=-1,
    )


@configclass
class ObjectYawVariationCfg(VariationBaseCfg):
    """Configuration for ObjectYawVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[-torch.pi / 2], high=[torch.pi / 2])
    )
    """Uniform distribution over the yaw delta applied at reset [rad].

    A rectangular footprint has period pi in yaw and mirrors about zero, so ``[-pi/2, pi/2]`` already
    covers every distinct grasp geometry. Widen only if the object's symmetry differs.
    """


class ObjectYawVariation(RunTimeVariationBase):
    """Rotate a rigid object about world Z at reset by a sampled delta.

    Applied after the placement reset event — Arena composes variation events after
    ``placement_reset`` — so the delta lands on a freshly solved layout and cannot compound across
    resets.

    Args:
        asset_name: Scene-entity name of the target rigid object.
        cfg: Tunable parameters; override the yaw distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: ObjectYawVariationCfg

    def __init__(
        self,
        asset_name: str,
        cfg: ObjectYawVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else ObjectYawVariationCfg()
        name = name if name is not None else f"object_yaw_{asset_name}"
        super().__init__(cfg=cfg, name=name)
        self.asset_name = asset_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"ObjectYawVariation on '{self.asset_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        return (
            f"{self.asset_name}_yaw_variation",
            EventTermCfg(
                func=apply_object_yaw_from_sampler,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(self.asset_name), "sampler": self._sampler},
            ),
        )


class apply_object_yaw_from_sampler(ManagerTermBase):
    """Event term: rotate a rigid object about world Z by a sampler-drawn yaw delta."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sampler: ContinuousSampler = cfg.params["sampler"]
        assert tuple(sampler.shape_per_sample) == (1,), (
            "apply_object_yaw_from_sampler expects a sampler with shape_per_sample (1,) over yaw; "
            f"got {tuple(sampler.shape_per_sample)}."
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        sampler: ContinuousSampler,
    ):
        import warp as wp

        asset = env.scene[asset_cfg.name]
        yaw = sampler.sample(len(env_ids), env_ids=env_ids).to(env.device).reshape(-1)

        pose = wp.to_torch(asset.data.root_link_pose_w)[env_ids].clone()
        pose[:, 3:7] = rotate_quat_xyzw_about_z(pose[:, 3:7], yaw)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        # Match the placement event: a rotated object must not inherit residual motion.
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
