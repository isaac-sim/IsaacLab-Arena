# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 environments for the thirty short motions in the spatial-memory guide."""

from dataclasses import dataclass
from typing import Literal

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class G1MotionEnvironmentCfg(ArenaEnvironmentCfg):
    scene: str = "shelf_bay"
    """Workcell key from the G1 guide, e.g. shelf_bay or moving_carriage."""

    layout: Literal["A", "B"] = "A"
    """Position of the physical feature that disappears from the close camera view."""

    mapped: bool = False
    """Add an online point cloud from head-camera depth. Requires cameras."""


@register_environment
class G1MotionEnvironment(ArenaEnvironmentFactory[G1MotionEnvironmentCfg]):
    """Build a G1, a paired workcell and the physical completion criteria of its local motion."""

    name = "g1_spatial_motion"
    _legacy_argparse_cfg_type = G1MotionEnvironmentCfg

    def build(self, cfg):
        from math import ceil, sqrt

        from isaaclab.sim import GroundPlaneCfg
        from pxr import Usd, UsdGeom

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.spatial_memory.g1_adapter import G1Adapter
        from isaaclab_arena_environments.spatial_memory.workcells.bodies import BODIES
        from isaaclab_arena_environments.spatial_memory.workcells.layouts import PILOT_STARTS
        from isaaclab_arena_environments.spatial_memory.workcells.physics import workcell_usd
        from isaaclab_arena_environments.spatial_memory.workcells.task import MotionTask

        if cfg.scene not in BODIES:
            raise ValueError(f"Unknown workcell {cfg.scene!r}. Available workcells: {', '.join(BODIES)}")
        if cfg.layout not in ("A", "B") or (cfg.mapped and not cfg.enable_cameras):
            raise ValueError("Choose layout A or B and enable cameras when requesting the map.")
        layout = int(cfg.layout == "B")
        _, path = workcell_usd(cfg.scene, layout)
        background = Object(name="workcell", usd_path=path, object_type=ObjectType.BASE, initial_pose=Pose.identity())
        kitchen = cfg.scene in {"dishwasher", "hanging_pan", "hatch_transfer", "scoop_pour"}
        hdr = "kiara_interior_robolab" if kitchen else "empty_warehouse_robolab"
        assets = [
            background,
            self.asset_registry.get_asset_by_name("ground_plane")(spawner_cfg=GroundPlaneCfg(color=(0.30, 0.31, 0.30))),
            self.asset_registry.get_asset_by_name("light")(hdr=hdr, intensity=850),
        ]
        for body in BODIES[cfg.scene]:
            assets.append(
                ObjectReference(
                    name=body.name,
                    parent_asset=background,
                    prim_path=f"{background.prim_path}/Body_{body.name}",
                    object_type=ObjectType.RIGID,
                )
            )
        adapter = G1Adapter()
        embodiment = adapter.make_embodiment(cfg.enable_cameras)
        if cfg.scene in PILOT_STARTS:
            translation, orientation, _ = PILOT_STARTS[cfg.scene]
            start = Pose(position_xyz=translation, rotation_xyzw=orientation)
        else:
            start = Pose(position_xyz=(0.0, -1.05, 0.78), rotation_xyzw=(0.0, 0.0, sqrt(0.5), sqrt(0.5)))
        embodiment.set_initial_pose(start)
        if cfg.enable_cameras:
            camera = embodiment.camera_config.robot_head_cam
            camera.data_types = list(dict.fromkeys([*camera.data_types, "distance_to_image_plane"]))
            camera.update_latest_camera_pose = True
        stage = Usd.Stage.Open(path)
        mover = stage.GetPrimAtPath("/Workcell/Body_mover")
        animation = None
        if mover:
            animation = dict(
                times=list(mover.GetCustomDataByKey("motionTimes")),
                poses=[tuple(v) for v in mover.GetCustomDataByKey("motionPoses")],
                repeat=cfg.scene != "closing_door",
            )
        seam_points = []
        if cfg.scene == "seam_inspection":
            for i in range(4):
                prim = stage.GetPrimAtPath(f"/Workcell/Fixtures/Weld{i}")
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                length = UsdGeom.Cylinder(prim).GetHeightAttr().Get()
                samples = max(1, ceil(length / 0.04))
                for j in range(samples):
                    seam_points.append(tuple(transform.Transform((0.0, 0.0, length * (j / samples - 0.5)))))
            seam_points.append(tuple(transform.Transform((0.0, 0.0, length / 2))))
        task = MotionTask(cfg.scene, layout, cfg.mapped, animation, seam_points)

        def configure(env_cfg):
            env_cfg = adapter.configure_env_cfg(env_cfg)
            env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 6.0)
            if cfg.enable_cameras:
                env_cfg.num_rerenders_on_reset = max(1, env_cfg.num_rerenders_on_reset)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=embodiment, scene=Scene(assets=assets), task=task, env_cfg_callback=configure
        )
