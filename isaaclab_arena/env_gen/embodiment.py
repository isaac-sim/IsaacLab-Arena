"""DROID embodiment cfgs for env_factory.

Matches `isaaclab_arena/embodiments/droid/droid.py` (Arena's DROID setup —
same one used by `droid_tabletop_pick_and_place_environment.py`) with two
intentional differences:

  1. The Arena DroidSceneCfg ships an extra `stand` fixture
     (`stand_instanceable.usd` scaled 1.2×1.2×1.7 at `/Robot_Stand`, the
     aluminum-bar pedestal). GoalSpec scenes already include a `franka_table`
     fixture as the robot mount, so a second stand would double up. We skip
     the stand entity and keep the scene's `franka_table` visible as the
     pedestal.
  2. Robot `init_state.pos` is set to `(-0.087, 0, 0)` — the same (x, y)
     as the scene's `franka_table` prim (see `scene_gen/scenes/base_empty.usda`)
     — so the robot base visually sits on top of that pedestal.

Otherwise cfgs are identical to Arena's DROID: flattened Franka + Robotiq 2F-85
USD, absolute joint-position action on the arm + binary 0→open / 1→close
on `finger_joint`, proprio ObservationGroup, reset-to-DROID-home event.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR
from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
from isaaclab_arena.embodiments.droid.observations import (
    arm_joint_pos,
    ee_pos,
    ee_quat,
    gripper_pos,
)


# DROID's default home pose — 7 arm joints + 6 Robotiq gripper joints.
# Matches `DroidEventCfg.init_franka_arm_pose.params["default_pose"]` in
# `isaaclab_arena/embodiments/droid/droid.py`.
DROID_HOME_JOINT_POSE = [
    0.0,                # panda_joint1
    -1 / 5 * torch.pi,  # panda_joint2
    0.0,                # panda_joint3
    -4 / 5 * torch.pi,  # panda_joint4
    0.0,                # panda_joint5
    3 / 5 * torch.pi,   # panda_joint6
    0.0,                # panda_joint7
    0.0,                # finger_joint
    0.0,                # right_outer_knuckle_joint
    0.0,                # right_inner_finger_joint
    0.0,                # right_inner_finger_knuckle_joint
    0.0,                # left_inner_finger_knuckle_joint
    0.0,                # left_inner_finger_joint
]


# ----------------------------------------------------------------------------
# Robot articulation (Franka + Robotiq 2F-85, flattened into one USD)
# ----------------------------------------------------------------------------

DROID_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/robot_library/droid"
            f"/franka_robotiq_2f_85_flattened.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Matches RoboLab's `DroidCfg` exactly (see
        # `robolab-main/robolab/robots/droid.py:40` — `pos=(0, 0, 0)`).
        # The `franka_table` prim in the scene USD is placed at (-0.087, 0, 0)
        # with a 180° Z rotation so its authored mount point maps back to
        # world origin — the robot just spawns at origin and lands on the
        # pedestal. No manual XY offset needed.
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * torch.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * torch.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * torch.pi,
            "panda_joint7": 0.0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=17.0,
            damping=0.02,
            velocity_limit=1.0,
        ),
    },
)


# ----------------------------------------------------------------------------
# End-effector frame (Robotiq tool center)
# ----------------------------------------------------------------------------

_ee_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/DroidEEFrame")
_ee_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

EE_FRAME_CFG = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/robot/panda_link0",
    debug_vis=False,
    visualizer_cfg=_ee_marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/robot/panda_link0",
            name="end_effector",
            offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        ),
    ],
)


# ----------------------------------------------------------------------------
# Cameras — mirrors `robolab-main/robolab/variations/camera.py` 1:1
#   - WRIST_CAM_CFG: on Robotiq base_link (from Arena `DroidCameraCfg`)
#   - EXTERNAL_CAM_CFG: OverShoulderLeft (2.1 mm wide lens, 57 cm left / 66 cm up)
#   - EGO_WIDE_CAM_CFG: forward-facing chest cam
#   - EGO_MIRRORED_HIGH_CAM_CFG / EGO_MIRRORED_WIDE_CAM_CFG: wide-lens 3rd-person
#     from front (high + low vantages)
#   - EGO_MIRRORED_CAM_CFG: narrow 24 mm lens 3rd-person, portrait 480×864
# Resolutions and intrinsics match RoboLab so the rendered images line up.
# ----------------------------------------------------------------------------

_WIDE_SPAWN = sim_utils.PinholeCameraCfg(
    focal_length=2.1,
    focus_distance=28.0,
    horizontal_aperture=5.376,
    vertical_aperture=3.024,
)

WRIST_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.011, -0.031, -0.074),
        rot=(-0.420, 0.570, 0.576, -0.409),
        convention="opengl",
    ),
)

# RoboLab's OverShoulderLeftCameraCfg — external_cam (left)
EXTERNAL_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/external_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_WIDE_SPAWN,
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.57, 0.66),
        rot=(-0.393, -0.195, 0.399, 0.805),
        convention="opengl",
    ),
)

# Over-shoulder RIGHT mirror of `EXTERNAL_CAM_CFG`. Not in RoboLab's
# `variations/camera.py` but present in Arena's `DroidCameraCfg.external_camera_2`
# (real DROID rigs have both shoulder cams). Y flipped from +0.57 to -0.57.
EXTERNAL_CAM_2_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/external_cam_2",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_WIDE_SPAWN,
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, -0.57, 0.66),
        rot=(0.805, 0.399, -0.195, -0.393),
        convention="opengl",
    ),
)

# RoboLab's EgocentricWideAngleCameraCfg — forward from robot chest
EGO_WIDE_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_wide_angle_camera",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_WIDE_SPAWN,
    offset=CameraCfg.OffsetCfg(
        pos=(0.15, 0.0, 0.5),
        rot=(0.653, 0.271, -0.271, -0.653),
        convention="opengl",
    ),
)

# RoboLab's EgocentricMirroredWideAngleHighCameraCfg
EGO_MIRRORED_HIGH_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_high_camera",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_WIDE_SPAWN,
    offset=CameraCfg.OffsetCfg(
        pos=(0.9, 0.0, 1.0),
        rot=(0.653, 0.271, 0.271, 0.653),
        convention="opengl",
    ),
)

# RoboLab's EgocentricMirroredWideAngleCameraCfg
EGO_MIRRORED_WIDE_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_camera",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=_WIDE_SPAWN,
    offset=CameraCfg.OffsetCfg(
        pos=(0.9, 0.0, 0.5),
        rot=(0.653, 0.271, 0.271, 0.653),
        convention="opengl",
    ),
)

# RoboLab's EgocentricMirroredCameraCfg — narrow 24 mm lens, 480×864 portrait
EGO_MIRRORED_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_camera",
    height=480,
    width=864,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.0),
        rot=(0.653, 0.271, 0.271, 0.653),
        convention="opengl",
    ),
)


# ----------------------------------------------------------------------------
# Actions — absolute joint position for arm + binary 0→open / 1→close gripper
# ----------------------------------------------------------------------------

@configclass
class _BinaryFingerActionCfg(BinaryJointPositionActionCfg):
    """Same as `DroidAbsoluteJointPositionActionsCfg.gripper_action`:
    reinterprets the binary action so 0 = open, 1 = close (the Isaac Lab
    default is inverted).
    """

    class_type = BinaryJointPositionZeroToOneAction


@configclass
class DroidJointPositionActionsCfg:
    """Joint-position arm action + binary 0/1 gripper.

    `use_default_offset=True` means the action is an offset from the home
    joint pose — zero action ⇒ home pose, so the robot stays still when the
    policy emits zeros. 8-dim action total (7 arm + 1 gripper).
    """

    arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: _BinaryFingerActionCfg = _BinaryFingerActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": torch.pi / 4},
    )


@configclass
class DroidIKActionsCfg:
    """Differential-IK delta-pose arm action + binary 0/1 gripper.

    7-dim action total: [dx, dy, dz, drx_axis_angle, dry, drz, gripper].
    This is the format CuRobo's `action_from_pose` produces, so the CuRobo
    runner needs this variant. Matches Arena's
    `DroidDifferentialIKActionsCfg`.
    """

    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_link0",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls",
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    gripper_action: _BinaryFingerActionCfg = _BinaryFingerActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": torch.pi / 4},
    )


# ----------------------------------------------------------------------------
# Observations — proprio group (arm joints, gripper 0..1, ee pose, last action)
# ----------------------------------------------------------------------------

@configclass
class DroidProprioObsGroup(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(func=gripper_pos)
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class DroidObservationsCfg:
    policy: DroidProprioObsGroup = DroidProprioObsGroup()


# ----------------------------------------------------------------------------
# Events — reset to DROID home pose + small Gaussian joint noise
# ----------------------------------------------------------------------------

@configclass
class DroidEventsCfg:
    """Reset-mode events.

    Only the deterministic `set_default_joint_pose` is kept. Arena's
    `DroidEventCfg` also runs `randomize_joint_by_gaussian_offset(std=0.02)`
    but that makes the robot visibly jitter on every reset — unwanted for a
    playable baseline. Add it back here if you need per-episode init noise
    for training.
    """

    init_droid_arm_pose = EventTermCfg(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": DROID_HOME_JOINT_POSE},
    )


# ----------------------------------------------------------------------------
# Entry points consumed by env_factory.
#
# Names kept as `get_franka_*` so env_factory doesn't need a rename. The
# embodiment is DROID (Franka arm + Robotiq 2F-85 gripper), but the
# integration surface is the same.
# ----------------------------------------------------------------------------

def get_franka_scene_entities(include_cameras: bool = True) -> dict:
    """Returns the scene-level entities the embodiment contributes.

    Deliberately does NOT include the aluminum-bar stand from
    `DroidSceneCfg.stand` — the GoalSpec scenes' own `franka_table` fixture
    serves that role.

    With `include_cameras=True`, spawns all five RoboLab cameras from
    `robolab-main/robolab/variations/camera.py` + the Robotiq wrist cam.
    Rendering cost scales linearly with camera count; disable cameras or
    trim the list below if performance matters more than parity with
    RoboLab's recording setup.
    """
    entities: dict = {
        "robot": DROID_ROBOT_CFG,
        "ee_frame": EE_FRAME_CFG,
    }
    if include_cameras:
        entities["wrist_cam"] = WRIST_CAM_CFG
        entities["external_cam"] = EXTERNAL_CAM_CFG
        entities["external_cam_2"] = EXTERNAL_CAM_2_CFG
        entities["egocentric_wide_angle_camera"] = EGO_WIDE_CAM_CFG
        entities["egocentric_mirrored_wide_angle_high_camera"] = EGO_MIRRORED_HIGH_CAM_CFG
        entities["egocentric_mirrored_wide_angle_camera"] = EGO_MIRRORED_WIDE_CAM_CFG
        entities["egocentric_mirrored_camera"] = EGO_MIRRORED_CAM_CFG
    return entities


def get_franka_actions_cfg(action_type: str = "joint_pos"):
    """Return the action configclass for the requested control mode.

    - "joint_pos": 8-dim `[7 arm joint offsets, 1 gripper]` (default; for
      training and zero-action smoke tests).
    - "ik_delta":  7-dim `[dx, dy, dz, drx, dry, drz, gripper]` (for CuRobo
      motion planning, whose `action_from_pose` emits this format).
    """
    if action_type == "joint_pos":
        return DroidJointPositionActionsCfg()
    if action_type == "ik_delta":
        return DroidIKActionsCfg()
    raise ValueError(f"Unknown action_type '{action_type}' (joint_pos|ik_delta)")


def get_franka_observations_cfg() -> DroidObservationsCfg:
    return DroidObservationsCfg()


def get_franka_events_cfg() -> DroidEventsCfg:
    return DroidEventsCfg()
