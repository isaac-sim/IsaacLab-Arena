# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""供 policy runner 使用的 Newton 网格 SDF 碰撞外部环境。

该环境保留水壶的原始视觉模型，并从
``scene/fstylejug_a01/fstylejug_a01_solid_collision.usda`` 中独立、封闭的
``Collision`` 网格烘焙 Newton SDF；水壶会落入静态纸箱碰撞体。必须通过
``--presets newton`` 启动，
例如：

    .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_steps 240 --headless --presets newton \\
        --external_environment_class_path \\
        examples.external_env.newton_sdf_env:NewtonSdfEnvironment newton_sdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_JUG_COLLISION_USD_PATH = _REPOSITORY_ROOT / "scene/fstylejug_a01/fstylejug_a01_solid_collision.usda"
# This rigid-body wrapper references _JUG_COLLISION_USD_PATH as /Jug/Collision,
# while retaining the original textured visual meshes.
_JUG_RIGID_USD_PATH = _REPOSITORY_ROOT / "scene/fstylejug_a01/fstylejug_a01_visual_solid_physx.usda"
_CONTAINER_USD_PATH = _REPOSITORY_ROOT / "scene/open_cardboard_box_hydroelastic.usda"
_SDF_CACHE_DIR = _REPOSITORY_ROOT / "scene/tmp/newton_sdf_cache"

_TABLE_POSITION = (0.0, 0.0, -0.05)
_JUG_POSITION = (0.45, 0.22, 0.25)
_CONTAINER_POSITION = (0.45, 0.22, 0.02)
_CONTACT_MARGIN = 0.002
_CONTACT_GAP = 0.01


def _register_sdf_callback(sdf_resolution: int) -> None:
    """Register the Newton callback which turns imported mesh colliders into SDFs."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    def configure_newton_sdf(_payload) -> None:
        """Bake cached SDFs after USD ingestion and before ModelBuilder finalization."""
        builder = NewtonManager._builder
        assert builder is not None, "Newton MODEL_INIT 时未找到 ModelBuilder。"
        _SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        sdf_shape_count = 0
        for shape_index, geometry in enumerate(builder.shape_source):
            label = str(builder.shape_label[shape_index])
            if "FStyleJug" not in label and "TargetContainer" not in label:
                continue
            if not hasattr(geometry, "build_sdf"):
                continue

            builder.shape_margin[shape_index] = _CONTACT_MARGIN
            builder.shape_gap[shape_index] = _CONTACT_GAP
            geometry.build_sdf(
                max_resolution=sdf_resolution,
                narrow_band_range=(-0.04, 0.04),
                margin=_CONTACT_GAP,
                scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
                cache_dir=_SDF_CACHE_DIR,
            )
            assert geometry.sdf is not None, f"{label} 的 SDF 构造失败。"
            builder.shape_force_sdf[shape_index] = True
            sdf_shape_count += 1

        assert sdf_shape_count > 0, "Newton 未从水壶或纸箱导入可构造 SDF 的网格。"
        print(
            f"[INFO] Newton SDF 碰撞模型已就绪：{sdf_shape_count} 个网格 shape，resolution={sdf_resolution}",
            flush=True,
        )

    NewtonManager.register_callback(
        configure_newton_sdf,
        PhysicsEvent.MODEL_INIT,
        name="arena_external_newton_sdf",
    )


class NewtonSdfEnvironment(ExampleEnvironmentBase):
    """创建使用 Newton 和显式网格 SDF 碰撞的零动作 Arena 环境。"""

    name: str = "newton_sdf"

    def get_env(self, args_cli: argparse.Namespace):
        """Return the external Arena environment assembled from local USD assets."""
        assert args_cli.presets == "newton", "NewtonSdfEnvironment 必须使用 --presets newton 启动。"
        assert _JUG_COLLISION_USD_PATH.is_file(), f"缺少水壶 SDF 碰撞模型：{_JUG_COLLISION_USD_PATH}"
        assert _JUG_RIGID_USD_PATH.is_file(), f"缺少水壶刚体封装 USD：{_JUG_RIGID_USD_PATH}"
        assert _CONTAINER_USD_PATH.is_file(), f"缺少纸箱 USD：{_CONTAINER_USD_PATH}"

        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        _register_sdf_callback(args_cli.sdf_resolution)
        collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=_CONTACT_GAP, rest_offset=0.0)
        contact_material = sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        table = Object(
            name="table",
            object_type=ObjectType.RIGID,
            spawner_cfg=sim_utils.CuboidCfg(
                size=(3.2, 3.2, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=collision_props,
                physics_material=contact_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            ),
            initial_pose=Pose(position_xyz=_TABLE_POSITION),
        )
        jug = Object(
            name="FStyleJug",
            object_type=ObjectType.RIGID,
            usd_path=str(_JUG_RIGID_USD_PATH),
            initial_pose=Pose(position_xyz=_JUG_POSITION),
            spawn_cfg_addon={"collision_props": collision_props, "physics_material": contact_material},
        )
        target_container = Object(
            name="TargetContainer",
            object_type=ObjectType.BASE,
            usd_path=str(_CONTAINER_USD_PATH),
            initial_pose=Pose(position_xyz=_CONTAINER_POSITION),
            spawn_cfg_addon={"collision_props": collision_props, "physics_material": contact_material},
        )
        light = self.asset_registry.get_asset_by_name("light")()
        scene = Scene(assets=[table, jug, target_container, light])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=NoEmbodiment(),
            scene=scene,
            task=NoTask(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add SDF cooking options to the policy runner CLI."""
        parser.add_argument(
            "--sdf_resolution",
            type=int,
            default=256,
            help="Newton 网格 SDF 最长轴的最大分辨率。",
        )
