# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coupled build-time asset variation for the easy gear-mesh family."""

from __future__ import annotations

import json
from dataclasses import field
from pathlib import Path

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

from ..asset_factories import GEAR_MESH_ASSET_PATHS

_SURFACE_ORIGIN_ARENA = (0.0990017409436448, 0.0155650225211777)
_VARIANTS = json.loads(Path(__file__).with_name("variants.json").read_text())["variants"]


@configclass
class GearFamilyVariationCfg(VariationBaseCfg):
    """Configuration for paired gear and board tooth-count sampling."""

    enabled: bool = True
    tooth_counts: list[int] = field(default_factory=lambda: [16, 20, 24])
    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)


class GearFamilyVariation(BuildTimeVariationBase):
    """Select one matching gear/board family before placement is solved."""

    cfg: GearFamilyVariationCfg

    def __init__(self, board, gear, task, cfg: GearFamilyVariationCfg | None = None):
        super().__init__(cfg or GearFamilyVariationCfg(), name="gear_family")
        self._board = board
        self._gear = gear
        self._task = task

    @staticmethod
    def _set_usd(asset, path) -> None:
        asset.usd_path = str(path)
        asset.object_cfg.spawn.usd_path = str(path)
        asset.bounding_box = None

    @classmethod
    def _set_board_layout(cls, board, layout: str) -> None:
        cls._set_usd(board, GEAR_MESH_ASSET_PATHS["board"])
        variants = {"layout": layout}
        board.spawn_cfg_addon = {
            **getattr(board, "spawn_cfg_addon", {}),
            "variants": variants,
        }
        board.object_cfg.spawn.variants = variants

    def _realize_at_build_time(self) -> None:
        tooth_counts = [int(value) for value in self.cfg.tooth_counts]
        if not tooth_counts or any(value not in (16, 20, 24) for value in tooth_counts):
            raise ValueError("gear_family tooth_counts must contain only 16, 20, or 24")
        assert self.sampler is not None
        teeth = int(self.sampler.sample(num_samples=1, choices=tooth_counts)[0])

        self._set_usd(self._gear, GEAR_MESH_ASSET_PATHS[f"{teeth}t"])
        self._set_board_layout(self._board, f"board_{teeth}")
        self._task.set_gear_teeth(teeth)

        radius = 0.0025 * (teeth + 2) / 2.0
        x_half = 0.075 - radius - 0.008
        y_half = 0.135 - radius - 0.008
        limits = next(relation for relation in self._gear.get_relations() if relation.name == "position_limits_box")
        center_x, center_y = -0.0309982590563552, 0.0155650225211777
        limits.x_min, limits.x_max = center_x - x_half, center_x + x_half
        limits.y_min, limits.y_max = center_y - y_half, center_y + y_half


@configclass
class GearLayoutVariationCfg(VariationBaseCfg):
    """Configuration for selecting one exact generated upstream layout."""

    enabled: bool = True
    family: str = "pair"
    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)


class GearLayoutVariation(BuildTimeVariationBase):
    """Select a coupled pair/train board, gear set, and source-authored poses."""

    cfg: GearLayoutVariationCfg

    def __init__(self, board, gears, task, cfg: GearLayoutVariationCfg):
        super().__init__(cfg, name="gear_layout")
        self._board = board
        self._gears = tuple(gears)
        self._task = task

    @staticmethod
    def _relation(asset, name):
        return next(relation for relation in asset.get_relations() if relation.name == name)

    @classmethod
    def _set_pose(cls, asset, xy, yaw) -> None:
        position = cls._relation(asset, "at_position")
        position.x = _SURFACE_ORIGIN_ARENA[0] + float(xy[0])
        position.y = _SURFACE_ORIGIN_ARENA[1] + float(xy[1])
        cls._relation(asset, "rotate_around_solution").yaw_rad = float(yaw)

    def _realize_at_build_time(self) -> None:
        rows = {row["name"]: row for row in _VARIANTS if row["family"] == self.cfg.family}
        if self.cfg.family not in {"pair", "train"} or not rows:
            raise ValueError("gear_layout family must be 'pair' or 'train'")
        assert self.sampler is not None
        selected_variant = self.sampler.sample(num_samples=1, choices=tuple(rows))[0]
        row = rows[selected_variant]
        if len(row["gears"]) != len(self._gears):
            raise ValueError("selected gear layout does not match the scene gear count")

        GearFamilyVariation._set_board_layout(self._board, row["board_variant"])
        self._set_pose(self._board, row["board_xy"], row["board_yaw"])
        for asset, gear in zip(self._gears, row["gears"], strict=True):
            GearFamilyVariation._set_usd(asset, GEAR_MESH_ASSET_PATHS[f"{int(gear['teeth'])}t"])
            self._set_pose(asset, gear["xy"], gear["yaw"])
        self._task.configure_layout(row["stations"], row["target_offsets_xyz"])
        self.selected_variant = selected_variant
