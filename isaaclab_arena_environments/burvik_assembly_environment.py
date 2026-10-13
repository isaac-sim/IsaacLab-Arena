# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bimanual YAM workbench with the six Blender-authored BURVIK parts."""

from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

REPOSITORY = Path(__file__).resolve().parents[1]


@dataclass
class BurvikAssemblyEnvironmentCfg(ArenaEnvironmentCfg):
    action_mode: str = "ik_pose"
    enable_scene_camera: bool = False
    enable_top_camera: bool = False
    asset_directory: str = str(REPOSITORY / "artifacts/ikea_burvik")
    robot_usd_path: str = str(
        REPOSITORY / ".cache/tool_hanging_mdl_patch/assets/industrial__i2rt_yam/i2rt_yam_default.usda"
    )
    episode_length_s: float = 600.0


@register_environment
class BurvikAssemblyEnvironment(ArenaEnvironmentFactory[BurvikAssemblyEnvironmentCfg]):
    name = "burvik_assembly_yam"

    def build(self, cfg: BurvikAssemblyEnvironmentCfg):
        import json
        from scipy.spatial.transform import Rotation

        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg
        from isaaclab.managers import EventTermCfg

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.yam.yam import IndustrialBimanualYamEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
        from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.burvik_assembly.connections import (
            clear_connections,
            engage_seated_connections,
            reset_workbench,
        )
        from isaaclab_arena_environments.burvik_assembly.geometry import (
            ASSEMBLY_ORDER,
            BENCH_HEIGHT,
            MOUNTS,
            attachment_pose,
            initial_part_poses,
        )
        from isaaclab_arena_environments.burvik_assembly.task import InstallBurvikPartTask

        root = Path(cfg.asset_directory).resolve()
        assert Path(cfg.robot_usd_path).is_file(), f"Missing local YAM USD: {cfg.robot_usd_path}"
        specification = json.loads((root / "specification.json").read_text())
        attachments = {}
        for name in ASSEMBLY_ORDER:
            position, rotation = attachment_pose(name, specification)
            attachments[name] = {"position": position.tolist(), "quaternion": rotation.as_quat().tolist()}

        parts = []
        for name, (position, rotation) in initial_part_poses(robot_layout=cfg.action_mode != "joint_position").items():
            path = root / "assets" / f"{name}.usd"
            assert path.is_file(), f"Missing BURVIK part: {path}"
            parts.append(
                Object(
                    name=name,
                    object_type=ObjectType.RIGID,
                    usd_path=str(path),
                    initial_pose=Pose(tuple(map(float, position)), tuple(map(float, rotation.as_quat()))),
                )
            )

        def cuboid(name, position, size, color):
            is_bench = name == "workbench"
            asset = Object(
                name=name,
                object_type=ObjectType.RIGID if is_bench else ObjectType.BASE,
                initial_pose=Pose(position),
                spawner_cfg=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00015, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if is_bench else None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.5),
                ),
            )
            if is_bench:
                asset.disable_reset_pose()
            return asset

        mounts = dict(MOUNTS)
        assets = [cuboid("workbench", (0.0, 0.0, BENCH_HEIGHT - 0.035), (2.0, 1.5, 0.07), (0.14, 0.19, 0.22))]
        if cfg.action_mode != "joint_position":
            # Raised mounts cover both low tube pickups and the handle's swept volume.
            for side, (x, y, z) in mounts.items():
                mounts[side] = (x, y, z + 0.15)
                assets.append(cuboid(f"{side}_pedestal", (x, y, z + 0.06625), (0.14, 0.14, 0.1325), (0.2, 0.23, 0.25)))
            # Presentation rails leave room beneath the thin handle for the YAM fingertips.
            for index, x in enumerate((-0.491, -0.211)):
                assets.append(
                    cuboid(
                        f"handle_support_{index}",
                        (x, -0.20, BENCH_HEIGHT + 0.0175),
                        (0.022, 0.36, 0.035),
                        (0.24, 0.27, 0.29),
                    )
                )
        for index, (x, y) in enumerate(((-0.82, -0.58), (-0.82, 0.58), (0.82, -0.58), (0.82, 0.58))):
            assets.append(cuboid(f"bench_support_{index}", (x, y, 0.35), (0.06, 0.06, 0.70), (0.055, 0.07, 0.083)))
        assets += [cuboid("floor", (0.0, 0.0, -0.04), (20.0, 20.0, 0.08), (0.055, 0.072, 0.094)), *parts]
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(600.0)
        assets += [light, self.asset_registry.get_asset_by_name("directional_light")()]

        camera_rig = None
        if cfg.enable_top_camera:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            from isaaclab_arena.embodiments.yam.cameras import BimanualYamCameraCfg

            camera_rig = BimanualYamCameraCfg()
            camera_rig.set_robot_mount_positions(mounts["left"], mounts["right"])
            if not cfg.enable_cameras:
                camera_rig.left_wrist_camera = None
                camera_rig.right_wrist_camera = None
            # Raise the imported overhead rig to frame the full assembly, including the handle.
            camera_rig.top_camera.offset.pos = (0.0, 0.06, 2.60)
            camera_rig.top_camera.data_types = ["rgb"]
            camera_rig.top_camera.update_period = 1.0 / 24.0
            camera_rig.top_camera.renderer_cfg = IsaacRtxRendererCfg()

        embodiment = IndustrialBimanualYamEmbodiment(
            robot_usd_path=cfg.robot_usd_path,
            instanceable_robot_usd_path=cfg.robot_usd_path,
            left_mount_position=mounts["left"],
            right_mount_position=mounts["right"],
            enable_cameras=cfg.enable_cameras or cfg.enable_top_camera,
            camera_config=camera_rig,
            enable_ee_frames=True,
            action_mode=cfg.action_mode,
        )
        for side, yaw in (("left", -90), ("right", 90)):
            robot = getattr(embodiment.scene_config, f"{side}_robot")
            robot.init_state.rot = tuple(map(float, Rotation.from_euler("z", yaw, degrees=True).as_quat()))
            # Park the wrists beside the work area so scripted part lifts do not pass through them.
            robot.init_state.joint_pos["joint1"] = 1.5707963267948966 if cfg.action_mode == "joint_position" else 0.0
            if cfg.action_mode != "joint_position":
                for name, actuator in robot.actuators.items():
                    if name.startswith("arm_"):
                        actuator.stiffness = 3000.0
                        actuator.damping = 80.0
                robot.actuators["gripper"].stiffness = 20000.0
                robot.actuators["gripper"].damping = 120.0
                robot.actuators["gripper"].joint_effort_limit = 100.0

        task = CompositeTaskBase(
            subtasks=[
                InstallBurvikPartTask(name, attachments[name], final=name == "handle") for name in ASSEMBLY_ORDER
            ],
            episode_length_s=cfg.episode_length_s,
            task_description=(
                "Install BURVIK legs 1, 2, 3, 4, then the carry handle; leave the table upright and stable."
            ),
            subtasks_are_sequential=True,
            desired_subtask_success_state=[True] * 5,
        )
        scene = Scene(assets)
        connection_events = make_configclass(
            "BurvikConnectionsEventsCfg",
            [
                ("reset_workbench", EventTermCfg, EventTermCfg(func=reset_workbench, mode="reset")),
                (
                    "clear_burvik_connections",
                    EventTermCfg,
                    EventTermCfg(func=clear_connections, mode="reset", params={"attachments": attachments}),
                ),
                (
                    "engage_burvik_connections",
                    EventTermCfg,
                    EventTermCfg(
                        func=engage_seated_connections,
                        mode="interval",
                        interval_range_s=(0.0, 0.0),
                        is_global_time=True,
                        params={"attachments": attachments},
                    ),
                ),
            ],
        )()

        def configure(env_cfg):
            from isaaclab.sensors import ContactSensorCfg

            # One control step per physics step preserves narrow pin clearances during guidance.
            env_cfg.sim.dt = 1.0 / 240.0
            env_cfg.decimation = 1
            env_cfg.sim.render_interval = 10
            env_cfg.sim.physics.enable_ccd = True
            env_cfg.sim.physics.min_position_iteration_count = 128
            env_cfg.sim.physics.min_velocity_iteration_count = 16
            env_cfg.sim.physics.enable_external_forces_every_iteration = False
            env_cfg.sim.physics.solve_articulation_contact_last = True
            env_cfg.events = combine_configclass_instances("BurvikEventsCfg", connection_events, env_cfg.events)
            for name in ("tray", *ASSEMBLY_ORDER[:-1]):
                setattr(
                    env_cfg.scene,
                    f"{name}_bench_contact",
                    ContactSensorCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/{name}",
                        update_period=0.0,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/workbench"],
                    ),
                )
            env_cfg.viewer = ViewerCfg(eye=(1.65, -1.9, 2.25), lookat=(0.0, 0.0, 1.10), resolution=(1280, 720))
            if cfg.enable_scene_camera:
                from isaaclab.sensors import CameraCfg
                from isaaclab_physx.renderers import IsaacRtxRendererCfg
                from pxr import Gf

                eye, target = (1.65, -1.9, 2.25), (0.0, 0.0, 1.10)
                matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
                q = matrix.ExtractRotationQuat()
                env_cfg.scene.demo_camera = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/demo_camera",
                    height=720,
                    width=1280,
                    data_types=["rgb"],
                    update_period=1.0 / 24.0,
                    renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=35, horizontal_aperture=36, clipping_range=(0.02, 100)
                    ),
                    offset=CameraCfg.OffsetCfg(pos=eye, rot=(*q.GetImaginary(), q.GetReal()), convention="opengl"),
                )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=scene,
            embodiment=embodiment,
            task=task,
            env_cfg_callback=configure,
        )
