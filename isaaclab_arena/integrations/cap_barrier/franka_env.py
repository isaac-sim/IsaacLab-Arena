# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal B=1 DROID Arena environment for the CAP barrier smoke.

Import this module only after ``SimulationApp`` has started. Its construction path
uses Arena's normal scene/embodiment builder rather than a standalone Isaac Lab task.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from .grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_BIN_POSE,
    CAP_GROCERY_CAMERA_PROFILES,
    CAP_GROCERY_DROID_HOME,
    CAP_GROCERY_OBJECT_ASSET,
    CAP_GROCERY_OBJECT_POSE,
    CAP_GROCERY_SUPPORT_ASSET,
    CAP_GROCERY_SUPPORT_POSE,
    CAP_GROCERY_SUPPORT_SIZE,
)
from .joint_mapping import (
    DROID_FINGER_JOINT,
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    PANDA_ARM_JOINTS,
    droid_binary_gripper_action,
    make_droid_joint_mapping,
)


def _configure_cap_droid_embodiment(embodiment: Any, *, stand_spawn: Any, initial_gripper_closed: bool = False) -> None:
    # The reused Franka reset helper preserves only the final two articulation joints,
    # which are mimic joints on DROID rather than the commanded finger joint. This fixed
    # profile retains the state-writing reset event but removes its Gaussian offset.
    joint_reset = embodiment.event_config.randomize_franka_joint_state
    joint_reset.params["mean"] = 0.0
    joint_reset.params["std"] = 0.0
    arm_action = embodiment.action_config.arm_action
    arm_action.joint_names = list(PANDA_ARM_JOINTS)
    arm_action.preserve_order = True
    arm_action.scale = 1.0
    arm_action.offset = 0.0
    arm_action.use_default_offset = False
    embodiment.scene_config.stand.spawn = stand_spawn
    if initial_gripper_closed:
        # Start the physical finger joint at the closed endpoint so a single
        # open_gripper skill (all the CAP ROS connector exposes today) yields an
        # unambiguous closed->open transition. The DROID reset events set the pose
        # and override init_state, so the closed start must be written on the reset
        # event: init_franka_arm_pose.default_pose lists the finger joint at index 7
        # (seven arm joints precede it). init_state is patched too for consistency.
        # The production smoke keeps the open default (open->close->open sequence).
        embodiment.scene_config.robot.init_state.joint_pos[DROID_FINGER_JOINT] = DROID_GRIPPER_CLOSED_POSITION_RAD
        default_pose = embodiment.event_config.init_franka_arm_pose.params["default_pose"]
        if len(default_pose) <= 7:
            raise RuntimeError("DROID init_franka_arm_pose.default_pose is missing the finger joint slot")
        default_pose[7] = DROID_GRIPPER_CLOSED_POSITION_RAD


class FrankaSimulationAdapter:
    """Expose the narrow simulator surface consumed by ``ArenaLockstepManager``."""

    def __init__(self, environment: Any):
        import torch

        self._torch = torch
        self._environment = environment
        self._unwrapped = environment.unwrapped
        if self._unwrapped.num_envs != 1:
            raise RuntimeError(f"CAP barrier smoke requires B=1, got {self._unwrapped.num_envs}")
        self._robot = self._unwrapped.scene["robot"]
        self._joint_names = tuple(self._robot.joint_names)
        self.joint_mapping = make_droid_joint_mapping(self._joint_names)

        action_manager = self._unwrapped.action_manager
        active_terms = tuple(action_manager.active_terms)
        term_dimensions = tuple(action_manager.action_term_dim)
        if active_terms != ("arm_action", "gripper_action") or term_dimensions != (7, 1):
            raise RuntimeError(
                "CAP DROID profile requires action terms arm_action[7], gripper_action[1], "
                f"got {list(zip(active_terms, term_dimensions, strict=True))}"
            )
        arm_term = action_manager.get_term("arm_action")
        gripper_term = action_manager.get_term("gripper_action")
        self.joint_mapping.assert_action_order((*arm_term._joint_names, *gripper_term._joint_names))
        self._arm_slice = slice(0, 7)
        self._gripper_slice = slice(7, 8)
        self.last_physics_step_started_at_s: float | None = None
        self.physics_step_count = 0
        self.reset_count = 0

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    @staticmethod
    def _as_tensor(value):
        return value.torch if hasattr(value, "torch") else value

    def read_joint_state(self) -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
        position = self._as_tensor(self._robot.data.joint_pos)[0].detach().cpu().tolist()
        velocity = self._as_tensor(self._robot.data.joint_vel)[0].detach().cpu().tolist()
        # This is the best available smoke-level effort sample. Its parity with a
        # physical joint-state effort remains a World State contract question.
        effort = self._as_tensor(self._robot.data.applied_torque)[0].detach().cpu().tolist()
        return position, velocity, effort

    def arm_positions(self) -> tuple[float, ...]:
        return self.abi_positions()[:7]

    def gripper_position(self) -> float:
        return self.abi_positions()[7]

    def abi_positions(self) -> tuple[float, ...]:
        positions, _, _ = self.read_joint_state()
        return self.joint_mapping.to_abi_order(positions)

    def synchronize(self) -> None:
        device = self._torch.device(self._unwrapped.device)
        if device.type == "cuda":
            self._torch.cuda.synchronize(device)

    def step_position_targets(self, positions_in_abi_order: Sequence[float]) -> None:
        if len(positions_in_abi_order) != 8:
            raise ValueError(f"expected eight DROID targets, got {len(positions_in_abi_order)}")
        gripper_action = droid_binary_gripper_action(float(positions_in_abi_order[7]))
        action = self._torch.zeros(
            (1, self._unwrapped.action_manager.total_action_dim),
            device=self._unwrapped.device,
            dtype=self._as_tensor(self._robot.data.joint_pos).dtype,
        )
        action[:, self._arm_slice] = self._torch.as_tensor(
            positions_in_abi_order[:7],
            device=self._unwrapped.device,
            dtype=action.dtype,
        )
        action[:, self._gripper_slice] = gripper_action
        self.synchronize()
        self.last_physics_step_started_at_s = time.monotonic()
        self._environment.step(action)
        self.physics_step_count += 1

    def reset_without_physics_step(self) -> None:
        # ManagerBasedEnv.reset writes reset state and performs sim.forward(); it
        # does not advance physics, which is the required CR-20 fence behavior.
        self._environment.reset()
        self.reset_count += 1

    def close(self) -> None:
        self._environment.close()


def _configure_cap_camera(embodiment: Any, camera_profile: str) -> None:
    if camera_profile == "libero":
        from isaaclab_arena_environments.libero_cameras import LiberoDroidPerceptionCameraCfg

        camera_config = LiberoDroidPerceptionCameraCfg()
    elif camera_profile == "oblique":
        from isaaclab_arena_environments.maple_cameras import MapleDroidPerceptionCameraCfg

        camera_config = MapleDroidPerceptionCameraCfg()
    else:
        raise ValueError(
            f"unsupported CAP grocery camera profile {camera_profile!r}; expected one of {CAP_GROCERY_CAMERA_PROFILES}"
        )
    # The perception bridge serializes this sensor's current world pose. Keep
    # pose updates enabled for both profiles so a future camera variation cannot
    # silently leave RGB-D paired with initialization-time extrinsics.
    camera_config.exterior_cam.update_latest_camera_pose = True
    embodiment.camera_config = camera_config


def _configure_cap_grocery_embodiment(embodiment: Any) -> None:
    """Pin the proven grocery home while preserving the calibrated base pose."""
    robot_init = embodiment.scene_config.robot.init_state
    if tuple(robot_init.pos) != (0.0, 0.0, 0.0) or tuple(robot_init.rot) != (
        0.0,
        0.0,
        0.0,
        1.0,
    ):
        raise RuntimeError("CAP grocery scene requires the identity DROID base pinned by arena_droid_b1 calibration")
    embodiment.set_initial_joint_pose(initial_joint_pose=list(CAP_GROCERY_DROID_HOME))


def _make_cap_grocery_assets(registry: Any, sim_utils: Any) -> list[Any]:
    """Build the deterministic grocery assets without requiring a live USD stage."""
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    ground_plane = registry.get_asset_by_name("ground_plane")()
    light = registry.get_asset_by_name("light")()
    light.set_intensity(1500.0)

    support = registry.get_asset_by_name(CAP_GROCERY_SUPPORT_ASSET)()
    support.set_initial_pose(Pose(*CAP_GROCERY_SUPPORT_POSE))
    half_x = CAP_GROCERY_SUPPORT_SIZE[0] * 0.5
    half_y = CAP_GROCERY_SUPPORT_SIZE[1] * 0.5
    half_z = CAP_GROCERY_SUPPORT_SIZE[2] * 0.5
    support.bounding_box = AxisAlignedBoundingBox(
        (-half_x, -half_y, -half_z),
        (half_x, half_y, half_z),
    )
    # Keep the support invisible so the open-vocabulary detector cannot suppress
    # the grocery as a child of a large table detection. Collision remains active.
    support.object_cfg.spawn = sim_utils.CuboidCfg(
        size=CAP_GROCERY_SUPPORT_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        visible=False,
    )

    grocery = registry.get_asset_by_name(CAP_GROCERY_OBJECT_ASSET)()
    if grocery.object_type != ObjectType.RIGID or not {
        "object",
        "graspable",
        "food",
    }.issubset(grocery.tags):
        raise RuntimeError(f"{CAP_GROCERY_OBJECT_ASSET} must remain a dynamic graspable food object")
    grocery.set_initial_pose(Pose(*CAP_GROCERY_OBJECT_POSE))

    destination = registry.get_asset_by_name(CAP_GROCERY_BIN_ASSET)()
    if destination.object_type != ObjectType.RIGID or not {
        "object",
        "container",
    }.issubset(destination.tags):
        raise RuntimeError(f"{CAP_GROCERY_BIN_ASSET} must remain a dynamic container")
    destination.set_initial_pose(Pose(*CAP_GROCERY_BIN_POSE))

    return [ground_plane, light, support, destination, grocery]


def _make_cap_grocery_scene(registry: Any, sim_utils: Any):
    """Build the deterministic grocery scene description."""
    from isaaclab_arena.scene.scene import Scene

    return Scene(assets=_make_cap_grocery_assets(registry, sim_utils))


def _make_cap_environment(
    *,
    device: str,
    initial_gripper_closed: bool,
    enable_cameras: bool,
    grocery_scene: bool,
    camera_profile: str,
) -> FrankaSimulationAdapter:
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    registry = AssetRegistry()
    embodiment = registry.get_asset_by_name("droid_abs_joint_pos")(enable_cameras=enable_cameras)
    if enable_cameras:
        _configure_cap_camera(embodiment, camera_profile)
    elif camera_profile not in CAP_GROCERY_CAMERA_PROFILES:
        raise ValueError(
            f"unsupported CAP camera profile {camera_profile!r}; expected one of {CAP_GROCERY_CAMERA_PROFILES}"
        )
    # The DROID stand is not hosted on the public S3 Nucleus. Retain its scene
    # entry because the embodiment updates its pose, but replace the unavailable
    # USD with the same inert placeholder used by Arena's LIBERO DROID profile.
    stand_spawn = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), visible=False)
    if grocery_scene:
        _configure_cap_grocery_embodiment(embodiment)
    _configure_cap_droid_embodiment(embodiment, stand_spawn=stand_spawn, initial_gripper_closed=initial_gripper_closed)

    if grocery_scene:
        scene = _make_cap_grocery_scene(registry, sim_utils)
        environment_name = "CAP-Barrier-DROID-Grocery-To-Bin-B1-v0"
    else:
        ground_plane = registry.get_asset_by_name("ground_plane")()
        light = registry.get_asset_by_name("light")()
        scene = Scene(assets=[ground_plane, light])
        environment_name = "CAP-Barrier-DROID-B1-v0"

    def configure_profile(cfg):
        cfg.sim.dt = 0.005
        cfg.decimation = 1
        return cfg

    description = IsaacLabArenaEnvironment(
        name=environment_name,
        scene=scene,
        embodiment=embodiment,
        task=NoTask(),
        env_cfg_callback=configure_profile,
    )
    builder = ArenaEnvBuilder(
        description,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device),
    )
    environment, cfg = builder.make_registered_and_return_cfg()
    if cfg.sim.dt != 0.005 or cfg.decimation != 1:
        environment.close()
        raise RuntimeError(f"CAP profile timing mismatch: dt={cfg.sim.dt}, decimation={cfg.decimation}")
    return FrankaSimulationAdapter(environment)


def make_cap_franka_environment(
    *,
    device: str = "cuda:0",
    initial_gripper_closed: bool = False,
    enable_cameras: bool = False,
) -> FrankaSimulationAdapter:
    """Build the fixed, bare arena_droid_b1 smoke profile after Kit startup."""
    return _make_cap_environment(
        device=device,
        initial_gripper_closed=initial_gripper_closed,
        enable_cameras=enable_cameras,
        grocery_scene=False,
        camera_profile="libero",
    )


def make_cap_grocery_to_bin_environment(
    *,
    device: str = "cuda:0",
    initial_gripper_closed: bool = False,
    enable_cameras: bool = True,
    camera_profile: str = "libero",
) -> FrankaSimulationAdapter:
    """Build the calibrated grocery-to-bin scene on the arena_droid_b1 profile.

    The DROID base remains at identity to match the pinned CAP base calibration.
    Both camera profiles publish their live world pose, so switching the camera
    changes no static world-to-base calibration.
    """
    return _make_cap_environment(
        device=device,
        initial_gripper_closed=initial_gripper_closed,
        enable_cameras=enable_cameras,
        grocery_scene=True,
        camera_profile=camera_profile,
    )
