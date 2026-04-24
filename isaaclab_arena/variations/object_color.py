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

from dataclasses import field
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler import Sampler, UniformSampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg
from isaaclab_arena.variations.variation_registry import register_variation

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.scene.scene import Scene


@configclass
class ObjectColorVariationCfg(VariationBaseCfg):
    """Configuration for :class:`ObjectColorVariation`.

    The default ``sampler`` is a 3D :class:`UniformSamplerCfg` over the full
    ``[0, 1]^3`` RGB cube — aggressive but universally valid; users wanting
    subtler tints can override individual bounds (e.g.
    ``...sampler.low=[0.4,0.4,0.4]``) or replace the whole cfg via
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.set_sampler`.

    Attributes:
        mode: Event mode forwarded to :class:`EventTermCfg`. ``"reset"``
            resamples on every episode reset; ``"prestartup"`` picks a
            stable color per env for the whole run.
        mesh_name: Sub-mesh selector forwarded to
            :class:`isaaclab.envs.mdp.randomize_visual_color`. Empty string
            targets all meshes under the asset's prim.
        sampler: RGB distribution. Currently pinned to
            :class:`~isaaclab_arena.variations.sampler.UniformSamplerCfg`
            because :meth:`ObjectColorVariation.build_event_cfg` only knows how
            to translate uniform bounds into the ``colors`` dict that
            :class:`randomize_visual_color` expects; see
            :meth:`_sampler_to_colors_spec` for details. A tagged-union /
            config-group mechanism can be introduced later if other sampler
            kinds (e.g. discrete palette) become useful here.
    """

    mode: str = "reset"
    mesh_name: str = ""
    sampler: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0])
    )


@register_variation
class ObjectColorVariation(VariationBase):
    """Randomize an object's visual color per env.

    Emits a single :class:`EventTermCfg` bound to
    :class:`isaaclab.envs.mdp.randomize_visual_color`. The target asset's
    bound material is replaced with a fresh ``OmniPBR`` instance whose
    ``diffuse_color_constant`` is sampled (uniformly over RGB) from the
    variation's sampler. The sampler is built from
    :attr:`ObjectColorVariationCfg.sampler` at construction time, so calling
    :meth:`enable` alone is sufficient for reasonable behaviour; users can
    narrow or replace the distribution either at construction time (via the
    ``sampler`` kwarg, see below) or later via
    :meth:`~isaaclab_arena.variations.variation_base.VariationBase.set_sampler`,
    which accepts both :class:`~isaaclab_arena.variations.sampler.SamplerCfg`
    (cfg-driven, keeps :attr:`cfg` in sync) and :class:`Sampler` (imperative,
    does not touch :attr:`cfg`) inputs.

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
        cfg: Tunable parameters (``mode``, ``mesh_name``, ``sampler``).
            Defaults to an :class:`ObjectColorVariationCfg` with sensible
            reset-time, all-meshes, full-RGB-uniform defaults; callers only
            need to supply a cfg to override those.
        sampler: Optional override for the RGB distribution. Mirrors
            :meth:`~isaaclab_arena.variations.variation_base.VariationBase.set_sampler`:
            a :class:`UniformSamplerCfg` is built into a live sampler **and**
            written back onto ``self.cfg.sampler`` (declarative round-trip);
            a bare :class:`Sampler` is stored directly without touching
            ``self.cfg`` (imperative override). When both ``cfg.sampler`` and
            ``sampler`` are supplied the ``sampler`` kwarg wins — the cfg's
            sampler field is replaced for cfg-driven overrides or left as a
            stale record for imperative overrides. When ``sampler`` is
            ``None`` the sampler in ``cfg`` is used as-is.
    """

    name = "color"

    #: Narrow the base class annotation so static checkers know
    #: ``self.cfg.mode`` / ``self.cfg.mesh_name`` / ``self.cfg.sampler`` are
    #: available.
    cfg: ObjectColorVariationCfg

    def __init__(
        self,
        asset: ObjectBase,
        cfg: ObjectColorVariationCfg | None = None,
        sampler: Sampler | UniformSamplerCfg | None = None,
    ):
        super().__init__(cfg=cfg if cfg is not None else ObjectColorVariationCfg())
        self.asset_name = asset.name
        self.set_sampler(sampler if sampler is not None else self.cfg.sampler)

    def build_event_cfg(self, scene: Scene) -> tuple[str, EventTermCfg]:  # noqa: ARG002
        assert self._sampler is not None, (
            f"ObjectColorVariation on '{self.asset_name}' is enabled but no sampler is set; "
            "call .set_sampler(...) before building the env."
        )
        colors = self._sampler_to_colors_spec()
        event_name = f"{self.asset_name}_color_variation"
        event_cfg = EventTermCfg(
            func=mdp.randomize_visual_color,
            mode=self.cfg.mode,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "colors": colors,
                "mesh_name": self.cfg.mesh_name,
                "event_name": event_name,
            },
        )
        return event_name, event_cfg

    def _sampler_to_colors_spec(self) -> dict[str, tuple[float, float]]:
        """Translate ``self._sampler`` into the ``colors`` dict the event term expects.

        :class:`randomize_visual_color` accepts either a list of discrete
        RGB triples or a dict ``{"r": (low, high), "g": (...), "b": (...)}``
        of per-channel uniform ranges. We currently only produce the dict
        form from a 3D :class:`UniformSampler`; richer sampler types (e.g.
        a discrete choice sampler) can extend this mapping later.
        """
        assert isinstance(self._sampler, UniformSampler), (
            f"ObjectColorVariation currently only supports UniformSampler; got {type(self._sampler).__name__}. "
            "Discrete palette support (DiscreteChoiceSampler) is planned but not implemented."
        )
        assert tuple(self._sampler.event_shape) == (3,), (
            "ObjectColorVariation expects a 3D UniformSampler over RGB; got event_shape "
            f"{tuple(self._sampler.event_shape)}."
        )
        low = self._sampler.low.tolist()
        high = self._sampler.high.tolist()
        return {
            "r": (low[0], high[0]),
            "g": (low[1], high[1]),
            "b": (low[2], high[2]),
        }
