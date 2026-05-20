# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-asset visual color variation.

Implements the "replace the bound material" color randomization path
validated in ``isaaclab_arena/examples/compile_env_notebook.py`` —
originally a thin wrapper around :class:`isaaclab.envs.mdp.randomize_visual_color`,
now reimplemented locally as :class:`randomize_visual_color_from_sampler`
so an Arena :class:`~isaaclab_arena.variations.sampler.Sampler` drives the
RGB draw instead of an opaque Replicator RNG. Each cloned env ends up with
a distinct random flat color; the asset's original diffuse texture is
dropped (the in-place tint path remains a TODO, see
``2026_04_21_color_variation_status.md``).

Pulling the sampling step out of Replicator and into our :class:`Sampler`
gives the variation system access to the actual values drawn at run time,
which the sensitivity-analysis effort needs in order to record per-episode
input factors. The hook itself is not implemented here; see
:meth:`~isaaclab_arena.variations.sampler.Sampler.write_sample_to_ledger`.
"""

from __future__ import annotations

import re
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.version import compare_versions

from isaaclab_arena.variations.sampler import Sampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg
from isaaclab_arena.variations.variation_registry import register_variation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

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
            because the runtime event term
            :func:`randomize_visual_color_from_sampler` only knows how to
            apply per-channel RGB samples drawn from a 3D
            :class:`~isaaclab_arena.variations.sampler.UniformSampler`. A
            tagged-union / config-group mechanism can be introduced later if
            other sampler kinds (e.g. discrete palette) become useful here.
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
        event_name = f"{self.asset_name}_color_variation"
        event_cfg = EventTermCfg(
            func=randomize_visual_color_from_sampler,
            mode=self.cfg.mode,
            params={
                "asset_cfg": SceneEntityCfg(self.asset_name),
                "sampler": self._sampler,
                "mesh_name": self.cfg.mesh_name,
            },
        )

        # event_cfg = make_variation_event(
        #     func=mdp.randomize_visual_color,
        #     sampler=self._sampler,
        #     mode=self.cfg.mode,
        #     params={
        #         "asset_cfg": SceneEntityCfg(self.asset_name),
        #         "colors": colors,
        #         "mesh_name": self.cfg.mesh_name,
        #         "event_name": event_name,
        #     },
        # )

        # event_cfg = EventTermCfg(
        #     func=make_wrapper(mdp.randomize_visual_color),
        #     mode=self.cfg.mode,
        #     params={
        #         "func": mdp.randomize_visual_color,
        #         "asset_cfg": SceneEntityCfg(self.asset_name),
        #         "colors": colors,
        #         "mesh_name": self.cfg.mesh_name,
        #         "event_name": event_name,
        #     },
        # )
        return event_name, event_cfg


class randomize_visual_color_from_sampler(ManagerTermBase):
    """Randomize the visual color of bodies on an asset, sampling via an Arena :class:`Sampler`.

    Locally-owned variant of :class:`isaaclab.envs.mdp.randomize_visual_color`
    that delegates RGB sampling to an Arena
    :class:`~isaaclab_arena.variations.sampler.Sampler` instead of letting
    Replicator's internal RNG draw the values opaquely. The sample tensor is
    visible to Python on every call, which is what the sensitivity-analysis
    recording layer needs in order to log per-episode input factors. Setup
    (replicator extension load, material creation, prim binding) mirrors the
    upstream class so the visual result is identical when the sampler matches
    the bounds.

    Only the modern Replicator code path (``omni.replicator.core >= 1.12.4``)
    is supported: the legacy path builds a ``rep.distribution.uniform`` node
    inside an OmniGraph and samples there, which there is no way to replace
    with a Python-side :class:`Sampler` without forking the graph. This is
    asserted at term init.

    .. note::
        Like the upstream variant, randomization is applied to *all* envs on
        every call regardless of ``env_ids``; per-env subsetting on the
        Replicator side is still an open item upstream.

    .. note::
        Scene replication (:attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics`)
        must be ``False`` so each env owns its own material prim — same
        constraint as the upstream variant.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled (local: isaacsim only available with Kit)
        from isaacsim.core.utils.extensions import enable_extension  # noqa: PLC0415

        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep  # noqa: PLC0415

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: Sampler = cfg.params["sampler"]
        mesh_name: str = cfg.params.get("mesh_name", "")

        assert not env.cfg.scene.replicate_physics, (
            "Cannot randomize visual color with scene replication enabled. "
            "Set 'replicate_physics=False' on InteractiveSceneCfg."
        )

        # Match the upstream RGB-only contract: we'll be writing a (num_prims, 3)
        # array into 'diffuse_color_constant', so the sampler must produce 3D
        # samples. Higher-dim distributions (e.g. with alpha) can be added later
        # alongside an extension to the upstream attribute write.
        assert tuple(sampler.event_shape) == (3,), (
            "randomize_visual_color_from_sampler expects a sampler with event_shape (3,) over RGB; "
            f"got {tuple(sampler.event_shape)}."
        )

        asset = env.scene[asset_cfg.name]
        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        # TODO: Need to make it work for multiple meshes (mirrors upstream TODO).

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)
        assert compare_versions(version, "1.12.4") >= 0, (
            "randomize_visual_color_from_sampler requires omni.replicator.core >= 1.12.4 "
            f"(found {version}); the legacy OmniGraph sampling path cannot be driven by an Arena Sampler."
        )

        stage = env.sim.stage
        prims_group = rep.functional.get.prims(path_pattern=mesh_prim_path, stage=stage)
        num_prims = len(prims_group)

        for prim in prims_group:
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        # TODO: Should we specify the value when creating the material? (mirrors upstream TODO).
        self.material_prims = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        sampler: Sampler,
        mesh_name: str = "",
    ):
        import omni.replicator.core as rep  # noqa: PLC0415

        num_prims = len(self.material_prims)
        sample = sampler.sample(num_samples=num_prims)
        random_colors = sample.detach().cpu().numpy()
        print("random_colors", random_colors)
        rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


# def make_variation_event(func: Callable, sampler: Sampler, mode: str, params: dict) -> EventTermCfg:

    
#     return EventTermCfg(
#         func=func,
#         mode=mode,
#         params=params,
#     )



# def make_wrapper(func: Callable) -> Callable:
#     def wrapped(*args, **kwargs):
#         print("before")
#         result = func(*args, **kwargs)
#         print("after")
#         return result

#     # Preserve the original signature
#     import inspect

#     wrapped.__signature__ = inspect.signature(func)

#     return wrapped


# def wrap_callable_class(cls: type) -> type:
#     original_call = cls.__call__

#     def new_call(self, *args, **kwargs) -> Any:
#         print("before")
#         result = original_call(self, *args, **kwargs)
#         print("after")
#         return result

#     # Copy signature of __call__
#     import inspect

#     new_call.__signature__ = inspect.signature(original_call)

#     # Optional: preserve metadata
#     new_call.__name__ = original_call.__name__

#     # Create new class
#     class Wrapped(cls):
#         __call__ = new_call

#     return Wrapped

