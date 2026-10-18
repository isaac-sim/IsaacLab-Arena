# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A video-inspired laptop-opening and cup-transfer environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class LaptopCupEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the tabletop scene and its ordered success conditions."""

    embodiment: str = "franka_ik"
    episode_length_s: float = 30.0
    laptop_open_angle_deg: float = 90.0
    cup_destination_offset_xyz: tuple[float, float, float] = (-0.02, -0.28, 0.0)
    """Cup destination in the laptop base frame: +X toward the hinge, +Y to the user's left."""
    placement_tolerance_m: float = 0.055
    minimum_lift_height_m: float = 0.06
    teleop_device: str | None = None

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0.0
        assert 0.0 < self.laptop_open_angle_deg < 120.0
        assert self.embodiment == "franka_ik", "This layout and action configuration use the Franka IK embodiment"
        assert len(self.cup_destination_offset_xyz) == 3
        assert all(math.isfinite(value) for value in self.cup_destination_offset_xyz)
        assert self.cup_destination_offset_xyz[1] < -0.23, "The destination must clear the laptop's right edge"


@register_environment
class LaptopCupEnvironment(ArenaEnvironmentFactory[LaptopCupEnvironmentCfg]):
    """Build a Franka scene with a hinged laptop, ceramic cup, and wooden table."""

    name = "laptop_cup"
    _legacy_argparse_cfg_type = LaptopCupEnvironmentCfg

    def build(self, cfg: LaptopCupEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the scene with local generated assets and two sequential subtasks."""
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg
        from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
        from isaaclab.sensors import CameraCfg
        from pxr import Gf

        from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.configclass import make_configclass
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.assets.laptop_cup.generate_assets import generate_laptop_cup_assets
        from isaaclab_arena_environments.assets.laptop_cup.objects import Laptop, WoodenTable
        from isaaclab_arena_environments.laptop_cup_task import LaptopCupTask

        asset_paths = generate_laptop_cup_assets(get_arena_asset_cache_dir() / "laptop_cup")
        table = WoodenTable(asset_paths["table"], initial_pose=Pose(position_xyz=(0.50, 0.0, 0.74)))
        laptop = Laptop(asset_paths["laptop"], initial_pose=Pose(position_xyz=(0.50, 0.0, 0.741)))
        cup = Object(
            name="cup",
            usd_path=asset_paths["cup"],
            object_type=ObjectType.RIGID,
            initial_pose=Pose(position_xyz=(0.48, 0.28, 0.743)),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=1.0,
                ),
            },
        )
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.10, 0.0, 0.8755)))

        ground = self.asset_registry.get_asset_by_name("ground_plane")(
            spawner_cfg=sim_utils.GroundPlaneCfg(color=(0.35, 0.33, 0.30)),
        )
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(color=(0.90, 0.88, 0.83), intensity=700.0),
        )
        sunlight = self.asset_registry.get_asset_by_name("directional_light")(
            initial_pose=Pose(rotation_xyzw=(0.2706, -0.2706, 0.0, 0.9239)),
            spawner_cfg=sim_utils.DistantLightCfg(color=(1.0, 0.92, 0.80), intensity=1800.0, angle=12.0),
        )
        task = LaptopCupTask(
            laptop=laptop,
            cup=cup,
            table=table,
            cup_destination_offset_xyz=cfg.cup_destination_offset_xyz,
            openness_threshold=cfg.laptop_open_angle_deg / 120.0,
            episode_length_s=cfg.episode_length_s,
            placement_tolerance_m=cfg.placement_tolerance_m,
            minimum_lift_height_m=cfg.minimum_lift_height_m,
        )
        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()

        def configure_cameras(env_cfg):
            camera_eye = (-0.65, -1.15, 1.65)
            camera_target = (0.48, 0.0, 0.85)
            env_cfg.viewer = ViewerCfg(eye=camera_eye, lookat=camera_target, resolution=(1280, 720))
            env_cfg.sim.render_interval = env_cfg.decimation
            if cfg.enable_cameras:
                # The scene camera leaves the Franka's wrist camera and its variations intact.
                camera_transform = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*camera_eye), Gf.Vec3d(*camera_target), Gf.Vec3d(0.0, 0.0, 1.0)
                )
                camera_rotation = camera_transform.GetInverse().ExtractRotationQuat()
                env_cfg.scene.table_cam = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/TableCamera",
                    height=720,
                    width=1280,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=28.0,
                        horizontal_aperture=32.0,
                        clipping_range=(0.01, 10.0),
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=camera_eye,
                        rot=(*camera_rotation.GetImaginary(), camera_rotation.GetReal()),
                        convention="opengl",
                    ),
                )
                table_camera_observation = ObservationTermCfg(
                    func=mdp.image,
                    params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
                )
                camera_observations = env_cfg.observations.camera_obs
                camera_observation_type = make_configclass(
                    "LaptopCupCameraObservationsCfg",
                    [("table_cam_rgb", ObservationTermCfg, table_camera_observation)],
                    bases=(type(camera_observations),),
                )
                camera_observation_parameters = {
                    field.name: getattr(camera_observations, field.name)
                    for field in fields(camera_observations)
                    if field.init
                }
                env_cfg.observations.camera_obs = camera_observation_type(**camera_observation_parameters)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[ground, light, sunlight, table, laptop, cup]),
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=configure_cameras,
        )
