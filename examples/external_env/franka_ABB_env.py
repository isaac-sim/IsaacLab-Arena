# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka 操作水壶并装入纸箱的柔性装箱外部环境。

运行：
    PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
      --policy_type zero_action \
      --num_steps 20000 \
      --viz kit \
      --external_environment_class_path \
      examples.external_env.franka_ABB_env:FrankaAbbFlexiblePackingEnvironment \
      franka_abb_flexible_packing
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_JUG_COLLISION_USD_PATH = _REPOSITORY_ROOT / "scene/fstylejug_a01/fstylejug_a01_solid_collision.usda"
# The rigid-body wrapper references _JUG_COLLISION_USD_PATH as its Collision mesh.
_JUG_RIGID_USD_PATH = _REPOSITORY_ROOT / "scene/fstylejug_a01/fstylejug_a01_visual_solid_physx.usda"
_CONTAINER_USD_PATH = _REPOSITORY_ROOT / "scene/open_cardboard_box_hydroelastic.usda"
_JUG_SLOT_CENTERS = (
    (0.22, -0.55),
    (0.46, -0.55),
    (0.22, -0.32),
    (0.46, -0.32),
    (0.22, -0.09),
    (0.46, -0.09),
    (0.70, -0.55),
    (0.70, -0.32),
    (0.70, -0.09),
    (0.70, 0.14),
)
_JUG_SLOT_HALF_RANGE_M = 0.03


def _register_newton_visual_mesh_callback() -> None:
    """Make the jug's visual-only meshes visible in the Newton visualizer."""
    import newton
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    def enable_jug_visual_meshes(_payload) -> None:
        """Enable rendering after Newton imports the USD stage and before it finalizes the model."""
        builder = NewtonManager._builder
        assert builder is not None, "Newton MODEL_INIT 时未找到 ModelBuilder。"

        visible_shape_count = 0
        for shape_index, label in enumerate(builder.shape_label):
            # The composed USD keeps the textured meshes below ``Visual`` and
            # the watertight SDF proxy at the sibling ``Collision`` path.
            # Newton imports non-collider meshes without VISIBLE by default.
            if "/Visual/" not in str(label):
                continue
            builder.shape_flags[shape_index] |= int(newton.ShapeFlags.VISIBLE)
            visible_shape_count += 1

        assert visible_shape_count > 0, "Newton 未导入任何 jug 视觉 Mesh。"
        print(f"[INFO] 已为 Newton visualizer 启用 {visible_shape_count} 个 jug 视觉 Mesh。", flush=True)

    # ``get_env`` returns before MODEL_INIT is dispatched, so the callback must
    # not be stored as a weak reference to this nested function.
    NewtonManager.register_callback(
        enable_jug_visual_meshes,
        PhysicsEvent.MODEL_INIT,
        name="arena_franka_abb_enable_jug_visual_meshes",
        wrap_weak_ref=False,
    )


def _enable_absolute_ik_target_visualization(embodiment) -> None:
    """Show the processed Franka IK target pose as a world-space frame marker."""
    import isaaclab.utils.math as math_utils
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG

    class VisualizedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
        """Differential IK action that draws its absolute end-effector target."""

        def __init__(self, cfg, env) -> None:
            super().__init__(cfg, env)
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/FrankaIkActionTarget"
            marker_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
            self._target_marker = VisualizationMarkers(marker_cfg)

        def process_actions(self, actions) -> None:
            """Process the relative IK command and display its resulting absolute target."""
            super().process_actions(actions)
            target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
                self._asset.data.root_pos_w.torch,
                self._asset.data.root_quat_w.torch,
                self._ik_controller.ee_pos_des,
                self._ik_controller.ee_quat_des,
            )
            self._target_marker.visualize(translations=target_pos_w, orientations=target_quat_w)

    embodiment.action_config.arm_action.class_type = VisualizedDifferentialInverseKinematicsAction


def _enable_newton_franka_gravity_compensation(embodiment) -> None:
    """Keep Franka gravity enabled and route MuJoCo compensation through its actuators."""
    from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg, MujocoRigidBodyPropertiesCfg

    robot = embodiment.scene_config.robot
    assert robot is not None, "Franka embodiment 未配置 robot articulation。"
    existing_rigid_props = robot.spawn.rigid_props
    robot.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(
        rigid_body_enabled=existing_rigid_props.rigid_body_enabled,
        kinematic_enabled=existing_rigid_props.kinematic_enabled,
        disable_gravity=False,
        gravcomp=1.0,
    )
    robot.spawn.joint_drive_props = MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)


class FrankaAbbFlexiblePackingEnvironment(ExampleEnvironmentBase):
    """Build a Franka flexible-packing scene on a flat plane."""

    name: str = "franka_abb_flexible_packing"

    def get_env(self, args_cli: argparse.Namespace):
        """Create the flat-plane scene with ten randomized jugs and a static container target."""
        assert _JUG_COLLISION_USD_PATH.is_file(), f"缺少水壶 SDF 碰撞模型：{_JUG_COLLISION_USD_PATH}"
        assert _JUG_RIGID_USD_PATH.is_file(), f"缺少水壶刚体封装 USD：{_JUG_RIGID_USD_PATH}"
        assert _CONTAINER_USD_PATH.is_file(), f"缺少 container USD：{_CONTAINER_USD_PATH}"

        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        if getattr(args_cli, "presets", None) == "newton":
            _register_newton_visual_mesh_callback()
        collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)
        contact_material = sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        flat_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        jugs = []
        for index, (x_center, y_center) in enumerate(_JUG_SLOT_CENTERS):
            jugs.append(
                Object(
                    name=f"jug_{index:02d}",
                    object_type=ObjectType.RIGID,
                    usd_path=str(_JUG_RIGID_USD_PATH),
                    initial_pose=PoseRange(
                        position_xyz_min=(
                            x_center - _JUG_SLOT_HALF_RANGE_M,
                            y_center - _JUG_SLOT_HALF_RANGE_M,
                            0.25,
                        ),
                        position_xyz_max=(
                            x_center + _JUG_SLOT_HALF_RANGE_M,
                            y_center + _JUG_SLOT_HALF_RANGE_M,
                            0.25,
                        ),
                        rpy_min=(0.0, 0.0, -math.pi),
                        rpy_max=(0.0, 0.0, math.pi),
                    ),
                    spawn_cfg_addon={"collision_props": collision_props, "physics_material": contact_material},
                )
            )
        assert len(jugs) == 10, "柔性装箱场景必须初始化 10 个 jug。"
        container = Object(
            name="container",
            object_type=ObjectType.BASE,
            usd_path=str(_CONTAINER_USD_PATH),
            initial_pose=Pose(position_xyz=(0.5, 0.5, 0.0)),
            spawn_cfg_addon={"collision_props": collision_props, "physics_material": contact_material},
        )
        light = self.asset_registry.get_asset_by_name("light")()
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(enable_cameras=args_cli.enable_cameras)
        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        if getattr(args_cli, "presets", None) == "newton":
            _enable_newton_franka_gravity_compensation(embodiment)
        _enable_absolute_ik_target_visualization(embodiment)

        scene = Scene(assets=[flat_plane, *jugs, container, light, directional_light])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add the teleoperation device selection argument."""
        parser.add_argument(
            "--teleop_device",
            type=str,
            default="keyboard",
            choices=("keyboard", "spacemouse", "openxr"),
            help="Teleoperation device name.",
        )
