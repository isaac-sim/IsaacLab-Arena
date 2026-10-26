# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-base G1 apple-to-plate task on a reconstructed Bluefield background.

Use ``g1_fixed_base_upper_body`` for Pink IK teleoperation or
``g1_fixed_base_upper_body_joint`` for joint-target policies. The pelvis is fixed and
the lower body is locked; the calibrated layout and cameras match the real setup.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

TUNED_PICK_UP_OBJECT_NAME = "apple_01_objaverse_robolab"
TUNED_DESTINATION_NAME = "clay_plates_hot3d_robolab"

# Env-local table height, measured from the background's TableMesh collider.
TABLE_SURFACE_Z = 0.0
# Initial clearance above the table to avoid collider penetration.
TABLE_AIRGAP = 0.005

# Evaluation time limit; 50 Hz control gives 300 steps in six seconds.
EPISODE_LENGTH_S = 6.0

# Fixed pelvis position; a +90 degree yaw faces the robot toward the table.
ROBOT_XY = (0.60, -0.374)
# Floor alignment and measured pelvis height (meters).
GROUND_Z = -0.7375
PELVIS_HEIGHT_ABOVE_GROUND = 0.710
ROBOT_Z = GROUND_Z + PELVIS_HEIGHT_ABOVE_GROUND  # -0.0275

# Open-arm reset posture and finger friction for the Bluefield fixed-base G1.
G1_STATIC_OPEN_ARM_JOINT_POS: dict[str, float] = {
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.5,
    "right_shoulder_yaw_joint": -0.5,
}
G1_STATIC_FINGER_FRICTION_MATERIAL_PATH = "/World/Materials/g1_static_pick_place_high_friction_fingers"
G1_STATIC_FINGER_STATIC_FRICTION = 6.0
G1_STATIC_FINGER_DYNAMIC_FRICTION = 5.0
G1_STATIC_FINGER_PRIM_NAME_MARKERS: tuple[str, ...] = ("hand", "thumb", "index", "middle")

# Left shoulder-pitch position from the G1 URDF, transformed into the environment frame.
LEFT_SHOULDER_XY = (ROBOT_XY[0] - 0.1002, ROBOT_XY[1])
LEFT_SHOULDER_Z = ROBOT_Z + 0.2918

# Maximum shoulder-to-apple distance used by reset sampling (meters).
LEFT_ARM_REACH_M = 0.525

# Offset from the plate's USD origin to its dish center (meters).
PLATE_ORIGIN_TO_CENTER_XY = (-0.0132, 0.0192)

# Clearance: 0.155 m plate radius + 0.040 m apple radius + 0.050 m hand clearance.
PLATE_APPLE_MIN_SEPARATION_M = 0.245
# Maximum number of redraws for invalid apple poses.
APPLE_PLACEMENT_MAX_DRAWS = 100

# Dish-center bounds in front of the robot (env-local meters).
DESTINATION_CENTER_X_RANGE = (0.58, 0.62)
DESTINATION_CENTER_Y_RANGE = (0.06, 0.12)

# Apple sampling box; the reset event rejects poses outside reach or too close to the plate.
APPLE_SPAWN_XY_CENTER = (0.3575, 0.038)
APPLE_SPAWN_XY_RANGE = (0.1075, 0.070)


def _plate_origin_xy(dish_center_xy: tuple[float, float]) -> tuple[float, float]:
    """Convert a dish-center position to the plate's USD-origin position."""
    return (
        dish_center_xy[0] - PLATE_ORIGIN_TO_CENTER_XY[0],
        dish_center_xy[1] - PLATE_ORIGIN_TO_CENTER_XY[1],
    )


def _object_lifted_from_spawn(env, object_name: str, reference_height: float):
    """Require a 1 cm lift above the known env-local spawn height, including its air gap."""
    object_z = env.arena_world.get_position_w(object_name)[:, 2] - env.scene.env_origins[:, 2]
    return object_z > reference_height + 0.01


# Check that the configured ranges allow apple-to-plate separation.
assert (
    math.hypot(
        DESTINATION_CENTER_X_RANGE[0] - (APPLE_SPAWN_XY_CENTER[0] - APPLE_SPAWN_XY_RANGE[0]),
        DESTINATION_CENTER_Y_RANGE[0] - (APPLE_SPAWN_XY_CENTER[1] - APPLE_SPAWN_XY_RANGE[1]),
    )
    > PLATE_APPLE_MIN_SEPARATION_M
), "The placement ranges cannot satisfy PLATE_APPLE_MIN_SEPARATION_M."


# Optional recolored assets; missing files fall back to stock assets.
G1_RECOLORED_USD_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "assets" / "g1" / "g1_29dof_metal_material.usd"
)

WARM_PLATE_USD_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "assets" / "clay_plates" / "clay_plates_warm.usd"
)

# USD-origin height above the bottom face at the tuned scales. Re-measure when scales change.
_USD_ORIGIN_ABOVE_BOTTOM_M: dict[str, float] = {
    "apple_01_objaverse_robolab": 0.0225,  # at scale 0.01194
    "clay_plates_hot3d_robolab": 0.0,  # USD origin at plate bottom, so scale-independent
}
# Spawn scales for an approximately 8 cm apple and 31 cm plate.
# Measure using compute_local_bounding_box_from_usd; root USD scales are not preserved by the spawner.
_TUNED_SCALES: dict[str, tuple[float, float, float]] = {
    TUNED_PICK_UP_OBJECT_NAME: (0.01194, 0.01194, 0.01194),
    TUNED_DESTINATION_NAME: (1.0289, 1.0289, 1.0289),
}

# Physical mass of the scaled apple, independent of the source asset's mass properties.
APPLE_MASS_KG = 0.20


def _table_spawn_z(asset_name: str) -> float:
    """Return the env-local root height with the asset's bottom above the table by TABLE_AIRGAP."""
    if asset_name in _USD_ORIGIN_ABOVE_BOTTOM_M:
        return TABLE_SURFACE_Z + TABLE_AIRGAP + _USD_ORIGIN_ABOVE_BOTTOM_M[asset_name]
    warnings.warn(
        f"bluefield_g1_fixed_base_pick_and_place: no measured USD-origin offset for '{asset_name}'; "
        "spawning at the table surface with no compensation. Verify the asset's bottom face lands on the table.",
        stacklevel=2,
    )
    return TABLE_SURFACE_Z + TABLE_AIRGAP


def _asset_scale(asset_name: str) -> tuple[float, float, float]:
    """Return the tuned uniform scale for ``asset_name``, or 1.0 with a warning."""
    if asset_name in _TUNED_SCALES:
        return _TUNED_SCALES[asset_name]
    warnings.warn(
        f"bluefield_g1_fixed_base_pick_and_place: no measured scale for '{asset_name}'; "
        "spawning at scale=(1.0, 1.0, 1.0). Verify visually.",
        stacklevel=2,
    )
    return (1.0, 1.0, 1.0)


# Kit viewer position and target.
BLUEFIELD_VIEWER_LOOKAT = (*APPLE_SPAWN_XY_CENTER, _table_spawn_z(TUNED_PICK_UP_OBJECT_NAME) + 0.3)
BLUEFIELD_VIEWER_EYE = (0.4, 1.50, 0.40)


def _randomize_apple_and_plate(env, env_ids, apple_cfg, plate_cfg, apple_spawn_z, plate_spawn_z) -> None:
    """Sample upright poses with apple reach and plate-clearance constraints.

    Args:
        env: The environment being reset.
        env_ids: Environments to resample.
        apple_cfg: Scene entity of the pick-up object.
        plate_cfg: Scene entity of the destination.
        apple_spawn_z: Env-local apple root height, including the spawn clearance.
        plate_spawn_z: Env-local plate root height, including the spawn clearance.
    """
    if env_ids is None:
        return

    import torch

    device = env.device
    count = len(env_ids)

    def sample(low: float, high: float) -> torch.Tensor:
        return torch.rand(count, device=device) * (high - low) + low

    plate_x = sample(*DESTINATION_CENTER_X_RANGE)
    plate_y = sample(*DESTINATION_CENTER_Y_RANGE)

    def sample_apple_xy() -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            sample(center - half_range, center + half_range)
            for center, half_range in zip(APPLE_SPAWN_XY_CENTER, APPLE_SPAWN_XY_RANGE)
        )

    def rejected(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Identify apple poses that violate clearance or reach constraints."""
        shoulder_distance = torch.sqrt(
            (x - LEFT_SHOULDER_XY[0]) ** 2 + (y - LEFT_SHOULDER_XY[1]) ** 2 + (apple_spawn_z - LEFT_SHOULDER_Z) ** 2
        )
        return (torch.hypot(x - plate_x, y - plate_y) < PLATE_APPLE_MIN_SEPARATION_M) | (
            shoulder_distance > LEFT_ARM_REACH_M
        )

    # Replace rejected poses while preserving accepted samples.
    apple_x, apple_y = sample_apple_xy()
    for _ in range(APPLE_PLACEMENT_MAX_DRAWS):
        redraw = rejected(apple_x, apple_y)
        if not bool(redraw.any()):
            break
        redraw_x, redraw_y = sample_apple_xy()
        apple_x = torch.where(redraw, redraw_x, apple_x)
        apple_y = torch.where(redraw, redraw_y, apple_y)

    origins = env.scene.env_origins[env_ids]
    upright = torch.zeros(count, 4, device=device)
    upright[:, 3] = 1.0  # identity, (x, y, z, w) as everywhere else in Arena
    zero_velocity = torch.zeros(count, 6, device=device)

    for asset_cfg, position_xy, spawn_z in (
        (apple_cfg, (apple_x, apple_y), apple_spawn_z),
        (plate_cfg, (plate_x - PLATE_ORIGIN_TO_CENTER_XY[0], plate_y - PLATE_ORIGIN_TO_CENTER_XY[1]), plate_spawn_z),
    ):
        asset = env.scene[asset_cfg.name]
        position = torch.stack([*position_xy, torch.full_like(apple_x, spawn_z)], dim=-1) + origins
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([position, upright], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)


def _warmup_initial_reset(env, env_ids, control_steps: int) -> None:
    """Warm up physics once, restore the sampled state, then let normal reset finish."""
    if getattr(env, "_bluefield_initial_warmup_done", False):
        return
    assert control_steps > 0 and len(env_ids) == env.num_envs, "Warm-up requires the initial full reset."
    assert not env._physics_handles_decimation, "Warm-up currently requires standard PhysX stepping."

    initial_state = env.scene.get_state()
    for name, articulation in env.scene.articulations.items():
        articulation.set_joint_position_target_index(
            target=initial_state["articulation"][name]["joint_position"], env_ids=env_ids
        )
    for _ in range(control_steps):
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        env.sim.render()

    env.scene.reset(env_ids)
    env.scene.reset_to(initial_state, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env._bluefield_initial_warmup_done = True
    print(f"[Bluefield] Initial reset warmed up for {control_steps} control steps; initial state restored.", flush=True)


@dataclass
class BluefieldG1FixedBasePickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Bluefield fixed-base G1 apple-to-plate task."""

    enable_cameras: bool = False
    """Spawn the calibrated torso and wrist cameras."""
    object: str = TUNED_PICK_UP_OBJECT_NAME
    destination: str = TUNED_DESTINATION_NAME
    background: str = "bluefield_recon"
    """Registered reconstruction containing the table collider."""
    embodiment: str = "g1_fixed_base_upper_body"
    """Fixed-base G1 with Pink IK; select the joint-target variant for closed-loop policies."""
    teleop_device: str | None = None
    task_description: str = "move the apple to the plate"
    lock_waist: bool = True
    """Exclude waist joints from Pink IK; the fixed-base embodiment also locks the waist actuators."""
    light_intensity: float = 1500.0
    """Dome-light intensity."""
    hdr: str | None = None
    """Optional registered HDR map for the dome light."""
    add_directional_light: bool = True
    """Add a directional light alongside the dome light."""
    episode_length_s: float = EPISODE_LENGTH_S
    """Episode time limit in seconds."""
    initial_reset_warmup_steps: int = 30
    """Initial warm-up control steps before restoring the sampled state; zero disables warm-up."""


@register_environment
class BluefieldG1FixedBasePickAndPlaceEnvironment(
    ArenaEnvironmentFactory[BluefieldG1FixedBasePickAndPlaceEnvironmentCfg]
):
    """Build the Bluefield scene, fixed-base G1 and apple-to-plate task."""

    name: str = "bluefield_g1_fixed_base_pick_and_place"
    _legacy_argparse_cfg_type = BluefieldG1FixedBasePickAndPlaceEnvironmentCfg

    def build(self, cfg: BluefieldG1FixedBasePickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab import sim as sim_utils
        from isaaclab.envs.common import ViewerCfg
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        from isaaclab_arena.embodiments.g1.g1 import G1RealCalibratedCameraCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange
        from isaaclab_arena_environments.mdp.bluefield_camera_position_variation import BluefieldCameraPositionVariation

        class BluefieldPickAndPlaceTask(PickAndPlaceTask):
            def get_termination_cfg(self):
                """Track a lift from the known spawn height followed by upstream placement."""
                termination_cfg = super().get_termination_cfg()
                # Replace settling and lift only; keep upstream placement and progress tracking.
                criteria = termination_cfg.success[0]
                assert len(criteria.predicate_sequence) == 3, "Expected upstream settle/lift/place milestones."
                termination_cfg.success[0] = replace(
                    criteria,
                    predicate_sequence=[
                        partial(
                            _object_lifted_from_spawn,
                            object_name=self.pick_up_object.name,
                            reference_height=pick_up_object_z,
                        ),
                        *criteria.predicate_sequence[2:],
                    ],
                )
                return termination_cfg

        background = self.asset_registry.get_asset_by_name(cfg.background)()

        # Foreground objects use the reconstructed table collider.
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)(scale=_asset_scale(cfg.object))
        # Mass and recolors apply to these asset instances only.
        pick_up_object.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=APPLE_MASS_KG)
        destination = self.asset_registry.get_asset_by_name(cfg.destination)(scale=_asset_scale(cfg.destination))
        if WARM_PLATE_USD_PATH.is_file():
            destination.usd_path = str(WARM_PLATE_USD_PATH)
            destination.object_cfg.spawn.usd_path = str(WARM_PLATE_USD_PATH)
        else:
            warnings.warn(
                "bluefield_g1_fixed_base_pick_and_place: warm plate USD not found at "
                f"{WARM_PLATE_USD_PATH}; using the stock (pale) Nucleus plate asset.",
                stacklevel=2,
            )
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            lock_waist=cfg.lock_waist,
        )
        embodiment.set_finger_contact_friction(
            material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
            dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
            prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS,
        )

        if G1_RECOLORED_USD_PATH.is_file():
            embodiment.scene_config.robot.spawn.usd_path = str(G1_RECOLORED_USD_PATH)
        else:
            warnings.warn(
                "bluefield_g1_fixed_base_pick_and_place: recolored G1 USD not found at "
                f"{G1_RECOLORED_USD_PATH}; using the default (white) Nucleus G1 asset.",
                stacklevel=2,
            )

        # Use the calibrated torso and wrist camera rig.
        embodiment.camera_config = G1RealCalibratedCameraCfg()
        embodiment.add_variation(BluefieldCameraPositionVariation())

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = (
            self.asset_registry.get_asset_by_name("directional_light")() if cfg.add_directional_light else None
        )

        if cfg.teleop_device is not None:
            # Registers "spacemouse_g1_left_arm" and its retargeter entry for this embodiment.
            import isaaclab_arena_g1.teleop.g1_left_arm_spacemouse  # noqa: F401

            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Fixed pelvis pose with +90 degree yaw.
        embodiment.set_initial_pose(
            Pose(position_xyz=(ROBOT_XY[0], ROBOT_XY[1], ROBOT_Z), rotation_xyzw=(0.0, 0.0, 0.70711, 0.70711))
        )
        embodiment.set_joint_initial_pos(G1_STATIC_OPEN_ARM_JOINT_POS)
        apple_min = tuple(center - radius for center, radius in zip(APPLE_SPAWN_XY_CENTER, APPLE_SPAWN_XY_RANGE))
        apple_max = tuple(center + radius for center, radius in zip(APPLE_SPAWN_XY_CENTER, APPLE_SPAWN_XY_RANGE))
        pick_up_object_z = _table_spawn_z(cfg.object)
        pick_up_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(*apple_min, pick_up_object_z),
                position_xyz_max=(*apple_max, pick_up_object_z),
                rpy_min=(0.0, 0.0, 0.0),
                rpy_max=(0.0, 0.0, 0.0),
            )
        )
        destination_z = _table_spawn_z(cfg.destination)
        destination_min = _plate_origin_xy((DESTINATION_CENTER_X_RANGE[0], DESTINATION_CENTER_Y_RANGE[0]))
        destination_max = _plate_origin_xy((DESTINATION_CENTER_X_RANGE[1], DESTINATION_CENTER_Y_RANGE[1]))
        destination.set_initial_pose(
            PoseRange(
                position_xyz_min=(*destination_min, destination_z),
                position_xyz_max=(*destination_max, destination_z),
                rpy_min=(0.0, 0.0, 0.0),
                rpy_max=(0.0, 0.0, 0.0),
            )
        )
        # The coupled reset event replaces the independent pose events.
        pick_up_object.disable_reset_pose()
        destination.disable_reset_pose()

        def env_cfg_callback(env_cfg):
            # WBC and existing demonstrations require 50 Hz control.
            env_cfg = set_control_rate_50hz(env_cfg)
            # Refresh cameras after reset to avoid carrying over the previous episode's images.
            env_cfg.num_rerenders_on_reset = 1
            env_cfg.viewer = ViewerCfg(eye=BLUEFIELD_VIEWER_EYE, lookat=BLUEFIELD_VIEWER_LOOKAT, origin_type="env")
            assert env_cfg.events is not None, "Object placement requires an event configuration."
            env_cfg.events.bluefield_randomize_apple_and_plate = EventTermCfg(
                func=_randomize_apple_and_plate,
                mode="reset",
                params={
                    "apple_cfg": SceneEntityCfg(cfg.object),
                    "plate_cfg": SceneEntityCfg(cfg.destination),
                    "apple_spawn_z": pick_up_object_z,
                    "plate_spawn_z": destination_z,
                },
            )
            assert cfg.initial_reset_warmup_steps >= 0, "initial_reset_warmup_steps must be non-negative."
            if cfg.initial_reset_warmup_steps:
                # Append after placement: native reset clears managers and renders fresh observations afterward.
                env_cfg.events.bluefield_initial_reset_warmup = EventTermCfg(
                    func=_warmup_initial_reset,
                    mode="reset",
                    params={"control_steps": cfg.initial_reset_warmup_steps},
                )
            return env_cfg

        scene_assets = [background, pick_up_object, destination, light]
        if directional_light is not None:
            scene_assets.append(directional_light)
        scene = Scene(assets=scene_assets)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=BluefieldPickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                destination_object=destination,
                background_scene=background,
                episode_length_s=cfg.episode_length_s,
                task_description=cfg.task_description,
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
