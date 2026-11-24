# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A small pinned subset of the Lab heterogeneous RoboLab asset suite."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset_factory
from isaaclab_arena.assets.sources.base import UsdSource

REPOSITORY = "LiFanxing/HeteroTask"
REVISION = "f7eacb6db16e3cc4479c5f620686337edc4e209e"
ASSETS = ("av_chess_pawn", "av_apple", "bin_b04", "cubebox_a02", "table_oak")


@lru_cache(maxsize=None)
def _asset_path(asset_name: str) -> str:
    override = os.environ.get("VLA_BENCHMARKS_ASSETS_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                REPOSITORY,
                repo_type="dataset",
                revision=REVISION,
                allow_patterns=[f"objects/{asset_name}/**"],
            )
        )
    path = root / "objects" / asset_name / f"{asset_name}.usd"
    assert path.is_file(), f"Missing Lab asset: {path}"
    return str(path)


class LabExampleUsdSource(UsdSource):
    """Acquire one original USD package and its textures without modifying the source."""

    def __init__(self, asset_name: str):
        assert asset_name in ASSETS
        self.asset_name = asset_name

    def to_local_usd_path(self) -> str:
        return _asset_path(self.asset_name)


def _register(asset_name: str):
    @register_asset_factory(
        registry_name=f"runtime_lab_{asset_name}", object_type=ObjectType.RIGID, tags=("runtime_selection_example",)
    )
    def make(name: str = asset_name, kinematic: bool = False, **kwargs):
        from isaaclab.sim import UsdPhysicsRigidBodyCfg

        from isaaclab_arena.assets.object import Object

        spawn_addon = dict(kwargs.pop("spawn_cfg_addon", {}) or {})
        spawn_addon.setdefault("scale", None)
        if kinematic:
            spawn_addon["rigid_props"] = UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
        return Object(
            name=name,
            object_type=ObjectType.RIGID,
            usd_source=LabExampleUsdSource(asset_name),
            spawn_cfg_addon=spawn_addon,
            **kwargs,
        )


for _name in ASSETS:
    _register(_name)


@register_asset_factory(
    registry_name="runtime_lab_ground", object_type=ObjectType.BASE, tags=("runtime_selection_example",)
)
def make_ground(name="ground", **kwargs):
    from isaaclab.sim import GroundPlaneCfg

    from isaaclab_arena.assets.background import Background

    return Background(name=name, object_min_z=-0.5, reset_nested_physics=False, spawner_cfg=GroundPlaneCfg(), **kwargs)
