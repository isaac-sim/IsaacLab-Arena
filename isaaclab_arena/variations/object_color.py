# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-asset visual color variation.

Wraps :class:`isaaclab.envs.mdp.randomize_visual_color` — the "replace the
bound material" path validated in ``isaaclab_arena/examples/compile_env_notebook.py``.
Each cloned env ends up with a distinct random flat color; the asset's
original diffuse texture is dropped (the in-place tint path remains a TODO,
see ``2026_04_21_color_variation_status.md``).

Sampler support in this POC is limited to a 3D :class:`UniformSampler` over
RGB. :class:`~isaaclab.envs.mdp.randomize_visual_color` samples internally
from the ``colors`` dict we build here; the sampler's bounds are forwarded
as-is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.variations.sampler import Sampler, UniformSampler
from isaaclab_arena.variations.variation_base import VariationBase
from isaaclab_arena.variations.variation_registry import register_variation

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.scene.scene import Scene


@register_variation("color")
class ObjectColorVariation(VariationBase):
    """Randomize an object's visual color per env.

    Emits a single :class:`EventTermCfg` bound to
    :class:`isaaclab.envs.mdp.randomize_visual_color`. The target asset's
    bound material is replaced with a fresh ``OmniPBR`` instance whose
    ``diffuse_color_constant`` is sampled (uniformly over RGB) from the
    bounds of the provided :class:`UniformSampler`.

    Requirements:
        * ``scene.replicate_physics`` must be False (the Arena default).
          With replication on, all envs share one material and per-env
          randomization is impossible. :class:`randomize_visual_color`
          asserts this at construction time.

    Args:
        asset: The :class:`~isaaclab_arena.assets.object_base.ObjectBase`
            (or subclass) instance whose visual color will be varied. The
            asset's ``name`` is used to resolve the scene entity at event
            time, so the same instance must also be registered on the
            :class:`~isaaclab_arena.scene.scene.Scene`.
        sampler: Distribution over RGB triples. Currently restricted to
            a :class:`UniformSampler` with ``event_shape == (3,)``.
        mode: Event mode. ``"reset"`` resamples on every episode reset;
            ``"prestartup"`` picks a stable color per env for the whole run.
        mesh_name: Sub-mesh selector forwarded to
            :class:`randomize_visual_color`. Empty string targets all
            meshes under the asset's prim.
    """

    def __init__(
        self,
        asset: ObjectBase,
        sampler: Sampler,
        mode: str = "reset",
        mesh_name: str = "",
    ):
        self.asset = asset
        self.sampler = sampler
        self.mode = mode
        self.mesh_name = mesh_name

    def build_event_cfg(self, scene: Scene) -> tuple[str, EventTermCfg]:  # noqa: ARG002
        colors = self._sampler_to_colors_spec()
        event_name = f"{self.asset.name}_color_variation"
        event_cfg = EventTermCfg(
            func=mdp.randomize_visual_color,
            mode=self.mode,
            params={
                "asset_cfg": SceneEntityCfg(self.asset.name),
                "colors": colors,
                "mesh_name": self.mesh_name,
                "event_name": event_name,
            },
        )
        return event_name, event_cfg

    def _sampler_to_colors_spec(self) -> dict[str, tuple[float, float]]:
        """Translate ``self.sampler`` into the ``colors`` dict the event term expects.

        :class:`randomize_visual_color` accepts either a list of discrete
        RGB triples or a dict ``{"r": (low, high), "g": (...), "b": (...)}``
        of per-channel uniform ranges. We currently only produce the dict
        form from a 3D :class:`UniformSampler`; richer sampler types (e.g.
        a discrete choice sampler) can extend this mapping later.
        """
        assert isinstance(self.sampler, UniformSampler), (
            f"ObjectColorVariation currently only supports UniformSampler; got {type(self.sampler).__name__}. "
            "Discrete palette support (DiscreteChoiceSampler) is planned but not implemented."
        )
        assert tuple(self.sampler.event_shape) == (3,), (
            "ObjectColorVariation expects a 3D UniformSampler over RGB; got event_shape "
            f"{tuple(self.sampler.event_shape)}."
        )
        low = self.sampler.low.tolist()
        high = self.sampler.high.tolist()
        return {
            "r": (low[0], high[0]),
            "g": (low[1], high[1]),
            "b": (low[2], high[2]),
        }
