# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-asset visual color variation.

Replaces the asset's bound material with a fresh ``OmniPBR`` instance whose
``diffuse_color_constant`` is sampled per env by an Arena
:class:`~isaaclab_arena.variations.sampler.Sampler`. The asset's original
diffuse texture is dropped.
"""

from __future__ import annotations

import re
import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.version import compare_versions

from isaaclab_arena.variations.sampler import Sampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.scene.scene import Scene


@configclass
class ObjectColorVariationCfg(VariationBaseCfg):
    """Configuration for :class:`ObjectColorVariation`.

    Attributes:
        mode: Event mode forwarded to :class:`EventTermCfg`. ``"reset"`` resamples
            on every episode reset; ``"prestartup"`` picks a stable color per env.
        mesh_name: Sub-mesh selector. Empty string targets all meshes under the
            asset's prim.
        sampler: RGB distribution. Defaults to a 3D uniform over the full
            ``[0, 1]^3`` cube.
    """

    mode: str = "reset"
    mesh_name: str = ""
    sampler: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0])
    )


class ObjectColorVariation(VariationBase):
    """Randomize an object's visual color per env.

    Requires ``scene.replicate_physics=False`` (the Arena default) so each env
    owns its own material prim.

    Args:
        asset: The object whose visual color will be varied. Its ``name`` is
            used to resolve the scene entity at event time.
        cfg: Tunable parameters. Defaults to an :class:`ObjectColorVariationCfg`
            with full-RGB-uniform reset-time defaults.
        sampler: Optional override for the RGB distribution. If ``None``, the
            sampler in ``cfg`` is used.
    """

    name = "color"

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

    Variant of :class:`isaaclab.envs.mdp.randomize_visual_color` that delegates
    RGB sampling to a Python-side :class:`Sampler` so the drawn values are
    visible to the recording layer. Requires ``omni.replicator.core >= 1.12.4``
    and ``scene.replicate_physics=False``. Like the upstream variant,
    randomization is applied to all envs on every call regardless of ``env_ids``.
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
