# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation that removes an object from the scene with a given probability.

Typical use is thinning out distractor clutter, so a policy sees a different subset of
non-task objects from episode to episode.

Each resetting environment draws independently, so parallel envs disagree and an object that was
gone last episode can be back this one. Coming back relies on something else restoring the pose:
relation placement, or the object's own pose reset event. An object with neither stays away once
it has drawn "gone".

Parking is realized in a reset event rather than a spawn pose because relation placement rewrites
every non-anchor object's pose on reset, and per-object pose events restore their own. Variation
events are composed after both, so the park is what survives.
"""

from __future__ import annotations

import math
import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.agentic_environment_generation.authoring_metadata import AuthoringMetadata, ParameterMetadata
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.bernoulli_sampler import BernoulliSampler, BernoulliSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class ObjectDisappearVariationCfg(VariationBaseCfg):
    """Configuration for :class:`ObjectDisappearVariation`."""

    away_position_xyz: tuple[float, float, float] = (1000.0, 0.0, 0.0)
    """Env-local position to park a disappeared object at, far enough out to clear every env's cameras.

    Both extremes of this axis misbehave, so an override wants to stay in the middle. Below the
    ground collider the solver depenetrates the object and launches it back up into the scene; the
    ground is an infinite half-space, so there is no free space under it. Past ~100 km, float32
    coordinates are too coarse for contact resolution and the object jitters and sinks through
    instead of resting.
    """

    sampler_cfg: BernoulliSamplerCfg = field(default_factory=BernoulliSamplerCfg)
    """Probability that the object disappears, drawn per environment on every reset."""


def set_object_poses_with_probability(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    sampler: BernoulliSampler,
) -> None:
    """Reset event that moves the object to ``pose`` in the resetting envs whose draw says so."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    disappeared = torch.as_tensor(sampler.sample(num_samples=len(env_ids), env_ids=env_ids), device=env.device)
    away_env_ids = env_ids[disappeared]
    if len(away_env_ids) > 0:
        set_object_pose(env, away_env_ids, asset_cfg=asset_cfg, pose=pose)


class ObjectDisappearVariation(RunTimeVariationBase):
    """Remove an object from the scene with a per-env, per-reset probability.

    Args:
        asset_name: Scene-entity name of the target object. Holding the name rather than the object
            keeps the asset's variation list free of a back-reference, which ``cfg.validate()``
            would otherwise follow in circles.
        cfg: Tunable parameters. Defaults to a 50% chance of disappearing.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"disappear"``.
    """

    cfg: ObjectDisappearVariationCfg

    authoring_metadata = AuthoringMetadata(
        configuration={
            "away_position_xyz": ParameterMetadata(
                units="m", description="Finite XYZ parking position in the environment frame."
            ),
            "sampler_cfg.probability": ParameterMetadata(minimum=0, maximum=1),
        },
        constraints=("Use for optional objects; the task may prohibit removing required inventory.",),
        reset_semantics="Parks selected objects after placement; reappearance requires an ordinary pose reset.",
    )

    def __init__(
        self,
        asset_name: str,
        cfg: ObjectDisappearVariationCfg | None = None,
        name: str = "disappear",
    ):
        super().__init__(cfg=cfg if cfg is not None else ObjectDisappearVariationCfg(), name=name)
        self.asset_name = asset_name

    def validate_cfg(self) -> None:
        assert len(self.cfg.away_position_xyz) == 3 and all(
            math.isfinite(x) for x in self.cfg.away_position_xyz
        ), "Disappearance parking position must contain three finite coordinates."

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"ObjectDisappearVariation on '{self.asset_name}': sampler not set."
        return (
            f"{self.asset_name}_{self.name}",
            EventTermCfg(
                func=set_object_poses_with_probability,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.asset_name),
                    # Re-tupled because Hydra overrides arrive as lists.
                    "pose": Pose(position_xyz=tuple(self.cfg.away_position_xyz)),
                    "sampler": self._sampler,
                },
            ),
        )
