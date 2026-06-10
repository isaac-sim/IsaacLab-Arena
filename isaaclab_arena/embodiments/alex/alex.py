# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
import os
import re
import torch
import warp as wp
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import MISSING
from typing import List

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.pink_ik import DampingTaskCfg, LocalFrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab_teleop import XrCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose

# ---------------------------------------------------------------------------
# Paths
#
# Alex models (``alex_V1_description/``):
#   Option A (recommended for ability hands): mount the ihmc-alex-sdk *root*:
#       ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk
#     Auto-detects ``/models/alex-models`` and ``/models/alex-ros2/ihmc_hands_ros2``.
#   Option B (nubs only): mount alex-models alone:
#       ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk/alex-models
#     Lands at ``/models`` inside the container.
#   Option C: set ``ALEX_MODELS_DIR`` explicitly inside the container.
#
# Ability Hand models (``ihmc_hands_ros2`` package):
#   Resolved automatically when the SDK root is mounted (Option A).
#   Otherwise set ``ABILITY_HAND_MODELS_DIR`` to the mounted hands package root.
# ---------------------------------------------------------------------------
_ABILITY_HAND_LEFT_URDF = os.path.join("urdf", "abilityHand", "ability_hand_left_large.urdf")


def _has_alex_v1_description(models_dir: str) -> bool:
    return os.path.isdir(os.path.join(models_dir, "alex_V1_description"))


def _resolve_alex_models_dir() -> str:
    if explicit := os.environ.get("ALEX_MODELS_DIR"):
        return explicit
    # Docker -m mounts a host directory at /models. Accept either alex-models alone
    # or the full ihmc-alex-sdk root (alex-models is then /models/alex-models).
    for candidate in ("/models", "/models/alex-models"):
        if _has_alex_v1_description(candidate):
            return candidate
    return "/models"


def _resolve_ability_hand_models_dir(alex_models_dir: str) -> str:
    if explicit := os.environ.get("ABILITY_HAND_MODELS_DIR"):
        return explicit
    sdk_root = os.path.dirname(alex_models_dir)
    candidates = [
        os.path.join(sdk_root, "alex-ros2", "ihmc_hands_ros2"),
        "/ihmc_hands_ros2",
    ]
    for candidate in candidates:
        root = os.path.normpath(candidate)
        if os.path.isfile(os.path.join(root, _ABILITY_HAND_LEFT_URDF)):
            return root
    return os.path.normpath(candidates[0])


_ALEX_MODELS_DIR = _resolve_alex_models_dir()
_ABILITY_HAND_MODELS_DIR = _resolve_ability_hand_models_dir(_ALEX_MODELS_DIR)

# ---------------------------------------------------------------------------
# Constants copied verbatim from
#   isaaclab_assets/ihmc/robots/alex/alex.py
# Only ALEX_URDF_PATH is changed: relative → absolute via _ALEX_MODELS_DIR.
# ---------------------------------------------------------------------------
ALEX_URDF_PATH = _ALEX_MODELS_DIR + "/"
HANDS_PATH = ""  # not used for nub-forearm configuration

ALEX_V1 = "V1"
ALEX_V2 = "V2"

ALEX_NUBFOREARMS_PARTS = ["head", "leftUpperArm", "leftFixedForearm", "rightUpperArm", "rightFixedForearm"]

CONTROL_DT = 0.02
SIM_DT = 0.005
MIN_DELAY_DT = 0.004
MAX_DELAY_DT = 0.008

EFFORT_LIMIT_115 = 217.2
EFFORT_LIMIT_85 = 160.7
EFFORT_LIMIT_76 = 96.8
EFFORT_LIMIT_68 = 70.5
EFFORT_LIMIT_S = 25.0
EFFORT_LIMIT_ANKLE_Y = 129.0
EFFORT_LIMIT_ANKLE_X = 40.0

VELOCITY_LIMIT_115 = 9.3
VELOCITY_LIMIT_85 = 10.38
VELOCITY_LIMIT_76 = 9.72
VELOCITY_LIMIT_68 = 10.59
VELOCITY_LIMIT_S = 17.3
VELOCITY_LIMIT_ANKLE_Y = VELOCITY_LIMIT_76
VELOCITY_LIMIT_ANKLE_X = VELOCITY_LIMIT_76

STIFFNESS_85_HIP_X = EFFORT_LIMIT_85 / 3.0
STIFFNESS_68_HIP_Z = EFFORT_LIMIT_68 / 1.5
STIFFNESS_115_HIP_Y = EFFORT_LIMIT_115 / 3.0
STIFFNESS_115_KNEE = EFFORT_LIMIT_115 / 3.0
STIFFNESS_76_ANKLE_Y = EFFORT_LIMIT_ANKLE_Y / 3.0
STIFFNESS_76_ANKLE_X = EFFORT_LIMIT_ANKLE_X / 3.0
STIFFNESS_85_SPINE_Z = EFFORT_LIMIT_85 / 3.0
STIFFNESS_S_NECK_Z = 5.0
STIFFNESS_S_NECK_Y = 5.0
STIFFNESS_85_SHOULDER_Y = EFFORT_LIMIT_85 / 9.0
STIFFNESS_85_SHOULDER_X = EFFORT_LIMIT_85 / 9.0
STIFFNESS_68_SHOULDER_Z = EFFORT_LIMIT_68 / 4.5
STIFFNESS_68_ELBOW_Y = EFFORT_LIMIT_68 / 4.5

DAMPING_85_HIP_X = EFFORT_LIMIT_85 / 20.0
DAMPING_68_HIP_Z = EFFORT_LIMIT_68 / 10.0
DAMPING_115_HIP_Y = EFFORT_LIMIT_115 / 20.0
DAMPING_115_KNEE = EFFORT_LIMIT_115 / 20.0
DAMPING_76_ANKLE_Y = EFFORT_LIMIT_ANKLE_Y / 20.0
DAMPING_76_ANKLE_X = EFFORT_LIMIT_ANKLE_X / 20.0
DAMPING_85_SPINE_Z = EFFORT_LIMIT_85 / 20.0
DAMPING_S_NECK_Z = 1.0
DAMPING_S_NECK_Y = 1.0
DAMPING_85_SHOULDER_Y = 8.0
DAMPING_85_SHOULDER_X = 8.0
DAMPING_68_SHOULDER_Z = 4.0
DAMPING_68_ELBOW_Y = 4.0

ARMATURE_SCALE = 1.0
ARMATURE_ANKLE_SCALE = 1.0
ARMATURE_85 = 0.062 * ARMATURE_SCALE
ARMATURE_68 = 0.020 * ARMATURE_SCALE
ARMATURE_115 = 0.167 * ARMATURE_SCALE
ARMATURE_76 = 0.037 * ARMATURE_SCALE
ARMATURE_ANKLE_X = ARMATURE_76 * ARMATURE_ANKLE_SCALE
ARMATURE_ANKLE_Y = ARMATURE_76 * ARMATURE_ANKLE_SCALE
ARMATURE_S = 0.005 * ARMATURE_SCALE


def merge_urdfs(
    robot_version: str,
    wanted_parts: List[str],
    fixed_joints: List[str] = [""],
    allowed_collisions: List[str] | None = None,
    output_name: str = "temp",
):
    """Assemble a URDF from per-part component files.

    Copied verbatim from isaaclab_assets/ihmc/robots/alex/alex.py;
    uses lxml so the output preserves the original formatting.
    """
    from lxml import etree

    prefix = "alex_"
    excluded_name = "IMU"

    main_directory = ALEX_URDF_PATH + prefix + robot_version + "_description/urdf/"

    base_urdf_file = main_directory + prefix + robot_version.lower() + ".lowerBody.urdf"
    base_urdf = etree.parse(base_urdf_file)
    base_robot = base_urdf.getroot()

    for joint in base_robot.findall("joint"):
        if excluded_name in joint.get("name", ""):
            base_robot.remove(joint)
    for link in base_robot.findall("link"):
        if excluded_name in link.get("name", ""):
            base_robot.remove(link)
    for gazebo in base_robot.findall("gazebo"):
        if excluded_name in gazebo.get("reference", ""):
            base_robot.remove(gazebo)

    for wanted_part in wanted_parts:
        directory = main_directory
        filename = prefix + robot_version.lower() + "." + wanted_part + ".urdf"
        curr_urdf = etree.parse(directory + filename).getroot()

        for joint in curr_urdf.findall("joint"):
            if excluded_name not in joint.get("name", ""):
                if joint.get("name", "") in fixed_joints:
                    joint.set("type", "fixed")
                base_robot.append(joint)

        for link in curr_urdf.findall("link"):
            if excluded_name not in link.get("name", ""):
                base_robot.append(link)

        for gazebo in curr_urdf.findall("gazebo"):
            if excluded_name not in gazebo.get("reference", ""):
                base_robot.append(gazebo)

    if allowed_collisions is not None:
        for link in base_robot.findall(".//link"):
            for collision in link.findall("collision"):
                if collision.get("name") not in allowed_collisions:
                    link.remove(collision)

    save_filepath = main_directory + output_name + ".urdf"
    base_urdf.write(save_filepath, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return save_filepath


def _resolve_mesh_paths(src_path: str, output_path: str) -> str:
    """Rewrite ``package://alex_V1_description/`` mesh paths to absolute paths.

    The merged URDF from merge_urdfs still uses package:// prefixes that the
    Isaac Sim USD importer cannot resolve without a ROS package index.
    """
    pkg_prefix = "package://alex_V1_description/"
    abs_prefix = os.path.join(_ALEX_MODELS_DIR, "alex_V1_description") + "/"
    marker = f"<!-- models_dir={_ALEX_MODELS_DIR} -->"

    if (
        os.path.exists(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(src_path)
        and marker in open(output_path).read(256)
    ):
        return output_path

    tree = ET.parse(src_path)
    root = tree.getroot()
    for el in root.iter():
        # Normalise any multi-space whitespace in attribute values.
        for attr, val in list(el.attrib.items()):
            normalised = re.sub(r"\s+", " ", val).strip()
            if normalised != val:
                el.set(attr, normalised)
        if el.tag == "mesh":
            fn = el.get("filename", "")
            if fn.startswith(pkg_prefix):
                el.set("filename", abs_prefix + fn[len(pkg_prefix):])

    tree.write(output_path, xml_declaration=True, encoding="unicode")
    with open(output_path, "a") as f:
        f.write(f"\n{marker}\n")
    return output_path


def _strip_collisions_for_pink_ik(src_path: str, output_path: str) -> str:
    """Write a collision-free URDF for Pinocchio / Pink IK.

    Alex URDFs use Isaac Sim ``<capsule>`` collision primitives that standard
    urdfdom cannot parse.  Pink IK only needs the kinematic tree, so collisions
    are removed rather than converted.
    """
    marker = f"<!-- pink_ik_kinematics_only source={src_path} -->"

    if (
        os.path.exists(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(src_path)
        and marker in open(output_path).read(512)
    ):
        return output_path

    tree = ET.parse(src_path)
    root = tree.getroot()
    for link in root.iter("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)

    tree.write(output_path, xml_declaration=True, encoding="unicode")
    with open(output_path, "a") as f:
        f.write(f"\n{marker}\n")
    return output_path


# Merged + resolved URDF paths (written once, cached by mtime).
_ALEX_URDF_DIR = os.path.join(_ALEX_MODELS_DIR, "alex_V1_description", "urdf")
_ALEX_MERGED_URDF_PATH = os.path.join(_ALEX_URDF_DIR, "alex_v1_nubs_arena.urdf")
_ALEX_RESOLVED_URDF_PATH = os.path.join(_ALEX_URDF_DIR, "alex_v1_nubs_arena_resolved.urdf")
_ALEX_PINK_IK_URDF_PATH = os.path.join(_ALEX_URDF_DIR, "alex_v1_nubs_arena_pink_ik.urdf")

# ---------------------------------------------------------------------------
# ArticulationCfg — copied verbatim from ALEX_V1_NUBS_DEFAULT_CFG in
#   isaaclab_assets/ihmc/robots/alex/alex.py
# asset_path is left empty here and filled in at embodiment __init__ time.
# ---------------------------------------------------------------------------
ALEX_V1_NUBS_DEFAULT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path="",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=[".*HIP_X", ".*HIP_Z", ".*HIP_Y", ".*KNEE_Y", ".*ANKLE_Y", ".*ANKLE_X"],
            stiffness={
                ".*HIP_X": STIFFNESS_85_HIP_X,
                ".*HIP_Z": STIFFNESS_68_HIP_Z,
                ".*HIP_Y": STIFFNESS_115_HIP_Y,
                ".*KNEE_Y": STIFFNESS_115_KNEE,
                ".*ANKLE_Y": STIFFNESS_76_ANKLE_Y,
                ".*ANKLE_X": STIFFNESS_76_ANKLE_X,
            },
            damping={
                ".*HIP_X": DAMPING_85_HIP_X,
                ".*HIP_Z": DAMPING_68_HIP_Z,
                ".*HIP_Y": DAMPING_115_HIP_Y,
                ".*KNEE_Y": DAMPING_115_KNEE,
                ".*ANKLE_Y": DAMPING_76_ANKLE_Y,
                ".*ANKLE_X": DAMPING_76_ANKLE_X,
            },
            velocity_limit_sim={
                ".*HIP_X": VELOCITY_LIMIT_85,
                ".*HIP_Z": VELOCITY_LIMIT_68,
                ".*HIP_Y": VELOCITY_LIMIT_115,
                ".*KNEE_Y": VELOCITY_LIMIT_115,
                ".*ANKLE_Y": VELOCITY_LIMIT_ANKLE_Y,
                ".*ANKLE_X": VELOCITY_LIMIT_ANKLE_X,
            },
            armature={
                ".*HIP_X": ARMATURE_85,
                ".*HIP_Z": ARMATURE_68,
                ".*HIP_Y": ARMATURE_115,
                ".*KNEE_Y": ARMATURE_115,
                ".*ANKLE_Y": ARMATURE_ANKLE_Y,
                ".*ANKLE_X": ARMATURE_ANKLE_X,
            },
            effort_limit_sim={
                ".*HIP_X": EFFORT_LIMIT_85,
                ".*HIP_Z": EFFORT_LIMIT_68,
                ".*HIP_Y": EFFORT_LIMIT_115,
                ".*KNEE_Y": EFFORT_LIMIT_115,
                ".*ANKLE_Y": EFFORT_LIMIT_ANKLE_Y,
                ".*ANKLE_X": EFFORT_LIMIT_ANKLE_X,
            },
        ),
        "torso": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=["SPINE_Z", "NECK_Z", "NECK_Y"],
            stiffness={
                "SPINE_Z": STIFFNESS_85_SPINE_Z,
                "NECK_Z": STIFFNESS_S_NECK_Z,
                "NECK_Y": STIFFNESS_S_NECK_Y,
            },
            damping={
                "SPINE_Z": DAMPING_85_SPINE_Z,
                "NECK_Z": DAMPING_S_NECK_Z,
                "NECK_Y": DAMPING_S_NECK_Y,
            },
            velocity_limit_sim={
                "SPINE_Z": VELOCITY_LIMIT_85,
                "NECK_Z": VELOCITY_LIMIT_S,
                "NECK_Y": VELOCITY_LIMIT_S,
            },
            armature={
                "SPINE_Z": ARMATURE_85,
                "NECK_Z": ARMATURE_S,
                "NECK_Y": ARMATURE_S,
            },
            effort_limit_sim={
                "SPINE_Z": EFFORT_LIMIT_85,
                "NECK_Z": EFFORT_LIMIT_S,
                "NECK_Y": EFFORT_LIMIT_S,
            },
        ),
        "arms": DelayedPDActuatorCfg(
            min_delay=math.floor(MIN_DELAY_DT / SIM_DT),
            max_delay=math.ceil(MAX_DELAY_DT / SIM_DT),
            joint_names_expr=[".*SHOULDER_Y", ".*SHOULDER_X", ".*SHOULDER_Z", ".*ELBOW_Y"],
            stiffness={
                ".*SHOULDER_Y": STIFFNESS_85_SHOULDER_Y,
                ".*SHOULDER_X": STIFFNESS_85_SHOULDER_X,
                ".*SHOULDER_Z": STIFFNESS_68_SHOULDER_Z,
                ".*ELBOW_Y": STIFFNESS_68_ELBOW_Y,
            },
            damping={
                ".*SHOULDER_Y": DAMPING_85_SHOULDER_Y,
                ".*SHOULDER_X": DAMPING_85_SHOULDER_X,
                ".*SHOULDER_Z": DAMPING_68_SHOULDER_Z,
                ".*ELBOW_Y": DAMPING_68_ELBOW_Y,
            },
            velocity_limit_sim={
                ".*SHOULDER_Y": VELOCITY_LIMIT_85,
                ".*SHOULDER_X": VELOCITY_LIMIT_85,
                ".*SHOULDER_Z": VELOCITY_LIMIT_68,
                ".*ELBOW_Y": VELOCITY_LIMIT_68,
            },
            armature={
                ".*SHOULDER_Y": ARMATURE_85,
                ".*SHOULDER_X": ARMATURE_85,
                ".*SHOULDER_Z": ARMATURE_68,
                ".*ELBOW_Y": ARMATURE_68,
            },
            effort_limit_sim={
                ".*SHOULDER_Y": EFFORT_LIMIT_85,
                ".*SHOULDER_X": EFFORT_LIMIT_85,
                ".*SHOULDER_Z": EFFORT_LIMIT_68,
                ".*ELBOW_Y": EFFORT_LIMIT_68,
            },
        ),
    },
)

# ---------------------------------------------------------------------------
# Arm joints used by the PINK IK action config
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES_LIST = [
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
]

# ---------------------------------------------------------------------------
# Ability Hand configuration
# ---------------------------------------------------------------------------
ALEX_ABILITY_HANDS_PARTS = [
    "head",
    "leftUpperArm",
    "leftForearm",
    "leftAbilityHandAdapter",
    "rightUpperArm",
    "rightForearm",
    "rightAbilityHandAdapter",
]

# Arm + wrist joints controlled by PINK IK (7 per arm = 14 total).
ARM_WRIST_JOINT_NAMES_LIST = [
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "LEFT_WRIST_Z",
    "LEFT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
    "RIGHT_GRIPPER_Z",
]

# Psyonic Ability Hand finger joints (10 per hand = 20 total).
# These are held passively at their default position; not IK-controlled.
ABILITY_HAND_JOINT_NAMES_LIST = [
    "left_ability_hand_index_q1",
    "left_ability_hand_index_q2",
    "left_ability_hand_middle_q1",
    "left_ability_hand_middle_q2",
    "left_ability_hand_ring_q1",
    "left_ability_hand_ring_q2",
    "left_ability_hand_pinky_q1",
    "left_ability_hand_pinky_q2",
    "left_ability_hand_thumb_q1",
    "left_ability_hand_thumb_q2",
    "right_ability_hand_index_q1",
    "right_ability_hand_index_q2",
    "right_ability_hand_middle_q1",
    "right_ability_hand_middle_q2",
    "right_ability_hand_ring_q1",
    "right_ability_hand_ring_q2",
    "right_ability_hand_pinky_q1",
    "right_ability_hand_pinky_q2",
    "right_ability_hand_thumb_q1",
    "right_ability_hand_thumb_q2",
]

# 4-bar linkage q2 lower limit at q1=0 is 0.766 rad; use a small margin so defaults
# pass PhysX limit checks (strict float comparison at the boundary fails).
_ABILITY_HAND_Q2_OPEN_POS = 0.77
# thumb_q1 ∈ [-1.74, 0]: 0 = closed, -1.74 = open; default to open.
_ABILITY_HAND_THUMB_Q1_OPEN_POS = -1.74
_ABILITY_HAND_DEFAULT_JOINT_POS = {
    joint: (
        _ABILITY_HAND_Q2_OPEN_POS
        if joint.endswith("_q2") and "thumb" not in joint
        else _ABILITY_HAND_THUMB_Q1_OPEN_POS
        if joint.endswith("thumb_q1")
        else 0.0
    )
    for joint in ABILITY_HAND_JOINT_NAMES_LIST
}

# Six independent joints per hand (q2 for digits 1–4 are URDF mimics of q1).
_ABILITY_HAND_INDEPENDENT_JOINT_SUFFIXES = [
    "index_q1",
    "middle_q1",
    "ring_q1",
    "pinky_q1",
    "thumb_q1",
    "thumb_q2",
]


def ability_hand_independent_joint_names(side: str) -> list[str]:
    return [f"{side}_ability_hand_{suffix}" for suffix in _ABILITY_HAND_INDEPENDENT_JOINT_SUFFIXES]


def ability_hand_full_joint_names(side: str) -> list[str]:
    return [joint for joint in ABILITY_HAND_JOINT_NAMES_LIST if joint.startswith(f"{side}_")]


# Left/right wrist pose keys emitted by Se3AbsRetargeter (7 floats each: xyz + quat xyzw).
ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS = [
    "l_pos_x",
    "l_pos_y",
    "l_pos_z",
    "l_quat_x",
    "l_quat_y",
    "l_quat_z",
    "l_quat_w",
]
ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS = [
    "r_pos_x",
    "r_pos_y",
    "r_pos_z",
    "r_quat_x",
    "r_quat_y",
    "r_quat_z",
    "r_quat_w",
]

# Hand joint layout for Pink IK ``actions[:, -num_hand_joints:]`` and ``hand_joint_names``.
# Interleaved left/right (same convention as GR1T2 pick-place teleop).
#   indices  0- 4: left finger q1 (index, middle, ring, pinky)
#   indices  5- 9: right finger q1
#   indices 10-11: thumb q1 (left, right)
#   indices 12-16: left finger q2
#   indices 17-21: right finger q2
#   indices 22-23: thumb q2 (left, right)
ABILITY_HAND_TELEOP_JOINT_ORDER = [
    "left_ability_hand_index_q1",
    "left_ability_hand_middle_q1",
    "left_ability_hand_ring_q1",
    "left_ability_hand_pinky_q1",
    "right_ability_hand_index_q1",
    "right_ability_hand_middle_q1",
    "right_ability_hand_ring_q1",
    "right_ability_hand_pinky_q1",
    "left_ability_hand_thumb_q1",
    "right_ability_hand_thumb_q1",
    "left_ability_hand_index_q2",
    "left_ability_hand_middle_q2",
    "left_ability_hand_ring_q2",
    "left_ability_hand_pinky_q2",
    "right_ability_hand_index_q2",
    "right_ability_hand_middle_q2",
    "right_ability_hand_ring_q2",
    "right_ability_hand_pinky_q2",
    "left_ability_hand_thumb_q2",
    "right_ability_hand_thumb_q2",
]

ALEX_ABILITY_HAND_WRIST_ACTION_DIM = len(ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS) + len(
    ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS
)
ALEX_ABILITY_HAND_HAND_ACTION_DIM = len(ABILITY_HAND_TELEOP_JOINT_ORDER)
ALEX_ABILITY_HAND_TOTAL_ACTION_DIM = ALEX_ABILITY_HAND_WRIST_ACTION_DIM + ALEX_ABILITY_HAND_HAND_ACTION_DIM
ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM = 10

# Indices within the 20-dim teleop hand block for each mimic per-eef gripper vector.
_ABILITY_HAND_LEFT_GRIPPER_TELEOP_INDICES = (0, 1, 2, 3, 8, 10, 11, 12, 13, 18)
_ABILITY_HAND_RIGHT_GRIPPER_TELEOP_INDICES = (4, 5, 6, 7, 9, 14, 15, 16, 17, 19)


def _pack_ability_hand_teleop_block(left_hand: torch.Tensor, right_hand: torch.Tensor) -> torch.Tensor:
    """Pack per-hand mimic gripper vectors into ``ABILITY_HAND_TELEOP_JOINT_ORDER`` layout."""
    assert left_hand.shape[-1] == ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM
    assert right_hand.shape[-1] == ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM
    if left_hand.ndim == 1:
        hand_block = torch.zeros(ALEX_ABILITY_HAND_HAND_ACTION_DIM, device=left_hand.device, dtype=left_hand.dtype)
        hand_block[list(_ABILITY_HAND_LEFT_GRIPPER_TELEOP_INDICES)] = left_hand
        hand_block[list(_ABILITY_HAND_RIGHT_GRIPPER_TELEOP_INDICES)] = right_hand
        return hand_block

    hand_block = torch.zeros(
        *left_hand.shape[:-1],
        ALEX_ABILITY_HAND_HAND_ACTION_DIM,
        device=left_hand.device,
        dtype=left_hand.dtype,
    )
    hand_block[..., list(_ABILITY_HAND_LEFT_GRIPPER_TELEOP_INDICES)] = left_hand
    hand_block[..., list(_ABILITY_HAND_RIGHT_GRIPPER_TELEOP_INDICES)] = right_hand
    return hand_block


def _unpack_ability_hand_gripper_actions(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a teleop action tensor into per-hand mimic gripper vectors."""
    hand_block = actions[..., ALEX_ABILITY_HAND_WRIST_ACTION_DIM :]
    return {
        "left": hand_block[..., list(_ABILITY_HAND_LEFT_GRIPPER_TELEOP_INDICES)],
        "right": hand_block[..., list(_ABILITY_HAND_RIGHT_GRIPPER_TELEOP_INDICES)],
    }


def build_alex_ability_hand_teleop_action_order() -> list[str]:
    """Canonical flattened teleop action order consumed by PinkInverseKinematicsAction.

    Layout: [left_wrist(7), right_wrist(7), hand_joints(20)] = 34 total.
    """
    return (
        ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS
        + ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS
        + ABILITY_HAND_TELEOP_JOINT_ORDER
    )

_ALEX_ABILITY_HANDS_MERGED_URDF_PATH = os.path.join(_ALEX_URDF_DIR, "alex_v1_ability_hands_arena.urdf")
_ALEX_ABILITY_HANDS_RESOLVED_URDF_PATH = os.path.join(_ALEX_URDF_DIR, "alex_v1_ability_hands_arena_resolved.urdf")
_ALEX_ABILITY_HANDS_PINK_IK_URDF_PATH = os.path.join(_ALEX_URDF_DIR, "alex_v1_ability_hands_arena_pink_ik.urdf")


def _merge_ability_hands_urdf(robot_version: str, output_name: str = "alex_v1_ability_hands_arena") -> str:
    """Assemble Alex with full forearms and Psyonic Ability Hands.

    Calls merge_urdfs for the base + forearm + adapter parts, then appends
    joints and links from the separate ability hand URDFs.
    """
    from lxml import etree

    temp_name = "_temp_" + output_name
    merged_path = merge_urdfs(robot_version, ALEX_ABILITY_HANDS_PARTS, output_name=temp_name)

    tree = etree.parse(merged_path)
    base_robot = tree.getroot()

    hand_urdf_dir = os.path.join(_ABILITY_HAND_MODELS_DIR, "urdf", "abilityHand")
    for hand_urdf_name in ["ability_hand_left_large.urdf", "ability_hand_right_large.urdf"]:
        hand_path = os.path.join(hand_urdf_dir, hand_urdf_name)
        assert os.path.isfile(hand_path), (
            f"Ability Hand URDF not found: {hand_path}\n"
            "Mount the ihmc-alex-sdk root (recommended):\n"
            "  ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk\n"
            "Or mount ihmc_hands_ros2 separately and set:\n"
            "  export ABILITY_HAND_MODELS_DIR=/ihmc_hands_ros2"
        )
        hand_root = etree.parse(hand_path).getroot()
        for child in hand_root:
            base_robot.append(child)

    save_filepath = os.path.join(_ALEX_URDF_DIR, output_name + ".urdf")
    tree.write(save_filepath, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return save_filepath


def _resolve_standalone_hand_urdf(side: str) -> str:
    """Resolve ``package://abilityHand/`` paths in a standalone hand URDF for dex-retargeting."""
    src_path = os.path.join(_ABILITY_HAND_MODELS_DIR, "urdf", "abilityHand", f"ability_hand_{side}_large.urdf")
    output_path = os.path.join(_ALEX_URDF_DIR, f"ability_hand_{side}_large_resolved.urdf")
    replacements = {
        "package://abilityHand/": os.path.join(_ABILITY_HAND_MODELS_DIR, "meshes", "abilityHand") + "/",
    }
    marker = f"<!-- hands_dir={_ABILITY_HAND_MODELS_DIR},side={side} -->"

    if (
        os.path.exists(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(src_path)
        and marker in open(output_path).read(512)
    ):
        return output_path

    tree = ET.parse(src_path)
    root = tree.getroot()
    for el in root.iter():
        if el.tag == "mesh":
            fn = el.get("filename", "")
            for pkg_prefix, abs_prefix in replacements.items():
                if fn.startswith(pkg_prefix):
                    el.set("filename", abs_prefix + fn[len(pkg_prefix):])
                    break

    tree.write(output_path, xml_declaration=True, encoding="unicode")
    with open(output_path, "a") as f:
        f.write(f"\n{marker}\n")
    return output_path


def _resolve_mesh_paths_ability_hands(src_path: str, output_path: str) -> str:
    """Rewrite package:// mesh paths to absolute paths for the ability-hands URDF.

    Resolves both ``package://alex_V1_description/`` and ``package://abilityHand/``.
    """
    replacements = {
        "package://alex_V1_description/": os.path.join(_ALEX_MODELS_DIR, "alex_V1_description") + "/",
        "package://abilityHand/": os.path.join(_ABILITY_HAND_MODELS_DIR, "meshes", "abilityHand") + "/",
    }
    marker = f"<!-- models_dir={_ALEX_MODELS_DIR},hands_dir={_ABILITY_HAND_MODELS_DIR} -->"

    if (
        os.path.exists(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(src_path)
        and marker in open(output_path).read(512)
    ):
        return output_path

    tree = ET.parse(src_path)
    root = tree.getroot()
    for el in root.iter():
        for attr, val in list(el.attrib.items()):
            normalised = re.sub(r"\s+", " ", val).strip()
            if normalised != val:
                el.set(attr, normalised)
        if el.tag == "mesh":
            fn = el.get("filename", "")
            for pkg_prefix, abs_prefix in replacements.items():
                if fn.startswith(pkg_prefix):
                    el.set("filename", abs_prefix + fn[len(pkg_prefix):])
                    break

    tree.write(output_path, xml_declaration=True, encoding="unicode")
    with open(output_path, "a") as f:
        f.write(f"\n{marker}\n")
    return output_path


# Alex URDF links live under ``Robot/Geometry/PELVIS_LINK/...`` after USD import.
# Instanced ``proto_asset_*`` prims block authoring cameras on those links, so ZED
# sensors spawn under ``Robot/`` and track ``HEAD_LINK`` each step via
# :func:`sync_alex_zed_cameras`.
_ALEX_ZED_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot"

# ---------------------------------------------------------------------------
# ZED X Mini bracket on Alex HEAD_LINK (forehead mount, 50 mm stereo baseline).
# Mount center xyz=(0.13041, -0.01079, 0.02381) aligns with the head mesh
# front face; pitched ~21° (rpy=(0, 0.3633, 0)) so the optical axis looks
# forward from the head, not along the link axis.
# ---------------------------------------------------------------------------
_ZED_STEREO_BASELINE_M = 0.050
_ZED_MOUNT_CENTER_XYZ = (0.13041, -0.01079, 0.02381)
_ZED_MOUNT_ROT_XYZW = (0.40, -0.40, -0.58, 0.58)
_ZED_LEFT_CAM_OFFSET = Pose(
    position_xyz=(
        _ZED_MOUNT_CENTER_XYZ[0],
        _ZED_MOUNT_CENTER_XYZ[1] + _ZED_STEREO_BASELINE_M / 2.0,
        _ZED_MOUNT_CENTER_XYZ[2],
    ),
    rotation_xyzw=_ZED_MOUNT_ROT_XYZW,
)
_ZED_RIGHT_CAM_OFFSET = Pose(
    position_xyz=(
        _ZED_MOUNT_CENTER_XYZ[0],
        _ZED_MOUNT_CENTER_XYZ[1] - _ZED_STEREO_BASELINE_M / 2.0,
        _ZED_MOUNT_CENTER_XYZ[2],
    ),
    rotation_xyzw=_ZED_MOUNT_ROT_XYZW,
)

# ZED X Mini 2.2 mm intrinsics at 640×480 (110° horizontal FOV).
_ZED_X_MINI_WIDTH = 640
_ZED_X_MINI_HEIGHT = 480
_ZED_X_MINI_FX = _ZED_X_MINI_WIDTH / (2.0 * math.tan(math.radians(110.0) / 2.0))
_ZED_X_MINI_INTRINSICS = [
    _ZED_X_MINI_FX,
    0.0,
    _ZED_X_MINI_WIDTH / 2.0,
    0.0,
    _ZED_X_MINI_FX,
    _ZED_X_MINI_HEIGHT / 2.0,
    0.0,
    0.0,
    1.0,
]
_ZED_X_MINI_PINHOLE_SPAWN = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
    intrinsic_matrix=_ZED_X_MINI_INTRINSICS,
    width=_ZED_X_MINI_WIDTH,
    height=_ZED_X_MINI_HEIGHT,
    focal_length=0.22,
    clipping_range=(0.1, 10.0),
)

_ALEX_XR_CFG = XrCfg(
    anchor_pos=(0.0, 0.0, -1.0),
    anchor_rot=(0.0, 0.0, -0.70711, 0.70711),
    anchor_prim_path="/World/envs/env_0/Robot/PELVIS_LINK",
    anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
    fixed_anchor_height=True,
)


@register_asset
class AlexPinkEmbodiment(EmbodimentBase):
    """Embodiment for the IHMC Alex V1 robot with PINK IK end-effector control.

    Requires the alex-models directory to be mounted into the container at /models:
        ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk/alex-models
    """

    name = "alex_pink"
    default_arm_mode = ArmMode.RIGHT

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        use_tiled_camera: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose)

        merged_urdf = merge_urdfs(ALEX_V1, ALEX_NUBFOREARMS_PARTS, output_name="alex_v1_nubs_arena")
        resolved_urdf = _resolve_mesh_paths(merged_urdf, _ALEX_RESOLVED_URDF_PATH)

        robot_cfg = copy.deepcopy(ALEX_V1_NUBS_DEFAULT_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.asset_path = resolved_urdf

        # Teleop needs high-stiffness arms to track IK targets precisely.
        # DelayedPDActuatorCfg uses physical gains (~18 Nm/rad) which lets arms sag.
        robot_cfg.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*"],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=3000.0,
            damping=100.0,
            armature=0.0,
        )

        # Start with elbows bent so IK has a natural initial configuration.
        robot_cfg.init_state.joint_pos = {
            "LEFT_ELBOW_Y": -1.5708,
            "RIGHT_ELBOW_Y": -1.5708,
        }

        self.scene_config = AlexSceneCfg()
        self.scene_config.robot = robot_cfg

        self.action_config = AlexActionsCfg()
        # PINK IK: kinematics-only URDF (Alex capsule collisions break urdfdom).
        pink_ik_urdf = _strip_collisions_for_pink_ik(merged_urdf, _ALEX_PINK_IK_URDF_PATH)
        self.action_config.upper_body_ik.controller.urdf_path = pink_ik_urdf
        self.action_config.upper_body_ik.controller.mesh_path = _ALEX_MODELS_DIR

        self.observation_config = AlexObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = AlexEventCfg()
        if enable_cameras:
            self.event_config.sync_zed_cameras = EventTerm(
                func=sync_alex_zed_cameras,
                mode="interval",
                interval_range_s=(CONTROL_DT, CONTROL_DT),
            )
            self.event_config.sync_zed_cameras_reset = EventTerm(
                func=sync_alex_zed_cameras,
                mode="reset",
            )

        self.mimic_env = AlexMimicEnv
        self.camera_config = AlexCameraCfg()
        self.camera_config._use_tiled_camera = use_tiled_camera
        self.camera_config.__post_init__()

        self.xr = copy.deepcopy(_ALEX_XR_CFG)


@configclass
class AlexSceneCfg:
    """Scene configuration for the Alex V1 robot."""

    robot: ArticulationCfg = MISSING


@configclass
class AlexActionsCfg:
    """Action configuration for Alex using PINK IK end-effector control."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=ARM_JOINT_NAMES_LIST,
        hand_joint_names=[],
        target_eef_link_names={
            "left": "LEFT_ELBOW_Y_LINK",
            "right": "RIGHT_ELBOW_Y_LINK",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="PELVIS_LINK",
            num_hand_joints=0,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                LocalFrameTaskCfg(
                    frame="LEFT_ELBOW_Y_LINK",
                    base_link_frame_name="PELVIS_LINK",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=75,
                    gain=0.075,
                ),
                LocalFrameTaskCfg(
                    frame="RIGHT_ELBOW_Y_LINK",
                    base_link_frame_name="PELVIS_LINK",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=75,
                    gain=0.075,
                ),
                DampingTaskCfg(cost=0.5),
                NullSpacePostureTaskCfg(cost=0.5),
            ],
        ),
    )


@configclass
class AlexCameraCfg:
    """ZED X Mini stereo camera configuration (left and right eyes)."""

    zed_left_cam: CameraCfg | TiledCameraCfg = MISSING
    zed_right_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        use_tiled = getattr(self, "_use_tiled_camera", False)
        CameraClass = TiledCameraCfg if use_tiled else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        common_kwargs = dict(
            update_period=0.0,
            update_latest_camera_pose=True,
            height=_ZED_X_MINI_HEIGHT,
            width=_ZED_X_MINI_WIDTH,
            data_types=["rgb"],
            spawn=_ZED_X_MINI_PINHOLE_SPAWN,
        )

        self.zed_left_cam = CameraClass(
            prim_path=_ALEX_ZED_CAMERA_PRIM_PATH + "/ZedLeftCam",
            offset=OffsetClass(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="opengl",
            ),
            **common_kwargs,
        )
        self.zed_right_cam = CameraClass(
            prim_path=_ALEX_ZED_CAMERA_PRIM_PATH + "/ZedRightCam",
            offset=OffsetClass(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="opengl",
            ),
            **common_kwargs,
        )


@configclass
class AlexObservationsCfg:
    """Observation specifications for Alex."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "LEFT_ELBOW_Y_LINK"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "LEFT_ELBOW_Y_LINK"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "RIGHT_ELBOW_Y_LINK"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "RIGHT_ELBOW_Y_LINK"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class AlexEventCfg:
    """Event configuration for Alex."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")

    sync_zed_cameras: EventTerm | None = None

    sync_zed_cameras_reset: EventTerm | None = None


def sync_alex_zed_cameras(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Track ZED cameras to the kinematic HEAD_LINK body each step."""
    if env_ids is None:
        return
    if "zed_left_cam" not in env.scene.sensors or "robot" not in env.scene.articulations:
        return

    robot = env.scene["robot"]
    head_body_ids, _ = robot.find_bodies(["HEAD_LINK"])
    head_idx = int(head_body_ids[0])
    head_pos = wp.to_torch(robot.data.body_pos_w)[env_ids, head_idx]
    head_quat = wp.to_torch(robot.data.body_quat_w)[env_ids, head_idx]

    for cam_name, offset in (
        ("zed_left_cam", _ZED_LEFT_CAM_OFFSET),
        ("zed_right_cam", _ZED_RIGHT_CAM_OFFSET),
    ):
        offset_pos = torch.tensor(offset.position_xyz, device=env.device, dtype=torch.float32).repeat(
            len(env_ids), 1
        )
        offset_quat = torch.tensor(offset.rotation_xyzw, device=env.device, dtype=torch.float32).repeat(
            len(env_ids), 1
        )
        cam_pos, cam_quat = PoseUtils.combine_frame_transforms(head_pos, head_quat, offset_pos, offset_quat)
        env.scene[cam_name].set_world_poses(cam_pos, cam_quat, env_ids, convention="opengl")


class AlexMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment for Alex — arms-only manipulation (no gripper)."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        rot_mat = PoseUtils.matrix_from_quat(eef_quat)
        return PoseUtils.make_pose(eef_pos, rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        left_pos, left_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        right_pos, right_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])
        left_quat = PoseUtils.quat_from_matrix(left_rot)
        right_quat = PoseUtils.quat_from_matrix(right_rot)

        if action_noise_dict is not None:
            left_pos += action_noise_dict["left"] * torch.randn_like(left_pos)
            right_pos += action_noise_dict["right"] * torch.randn_like(right_pos)
            left_quat += action_noise_dict["left"] * torch.randn_like(left_quat)
            right_quat += action_noise_dict["right"] * torch.randn_like(right_quat)

        return torch.cat((left_pos, left_quat, right_pos, right_quat), dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        left_pos = action[:, 0:3]
        left_rot = PoseUtils.matrix_from_quat(action[:, 3:7])
        right_pos = action[:, 7:10]
        right_rot = PoseUtils.matrix_from_quat(action[:, 10:14])
        return {
            "left": PoseUtils.make_pose(left_pos, left_rot),
            "right": PoseUtils.make_pose(right_pos, right_rot),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # Alex has no gripper — return empty tensors.
        empty = torch.zeros(actions.shape[0], 0, device=actions.device)
        return {"left": empty, "right": empty}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)


class AlexAbilityHandMimicEnv(AlexMimicEnv):
    """Mimic environment for Alex with Ability Hands (34-D Pink IK teleop actions)."""

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])
        target_left_eef_rot_quat = PoseUtils.quat_unique(PoseUtils.quat_from_matrix(left_target_rot))
        target_right_eef_rot_quat = PoseUtils.quat_unique(PoseUtils.quat_from_matrix(right_target_rot))

        left_hand_action = gripper_action_dict["left"]
        right_hand_action = gripper_action_dict["right"]

        if action_noise_dict is not None:
            target_left_eef_pos = target_left_eef_pos + action_noise_dict["left"] * torch.randn_like(target_left_eef_pos)
            target_right_eef_pos = target_right_eef_pos + action_noise_dict["right"] * torch.randn_like(
                target_right_eef_pos
            )
            target_left_eef_rot_quat = target_left_eef_rot_quat + action_noise_dict["left"] * torch.randn_like(
                target_left_eef_rot_quat
            )
            target_right_eef_rot_quat = target_right_eef_rot_quat + action_noise_dict["right"] * torch.randn_like(
                target_right_eef_rot_quat
            )

        hand_block = _pack_ability_hand_teleop_block(left_hand_action, right_hand_action)
        return torch.cat(
            (
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                hand_block,
            ),
            dim=-1,
        )

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return _unpack_ability_hand_gripper_actions(actions)


# ===========================================================================
# Alex with Psyonic Ability Hands
# ===========================================================================


@configclass
class AlexAbilityHandActionsCfg:
    """Action configuration for Alex with Psyonic Ability Hands — PINK IK wrist control."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=ARM_WRIST_JOINT_NAMES_LIST,
        hand_joint_names=ABILITY_HAND_TELEOP_JOINT_ORDER,
        target_eef_link_names={
            "left": "LEFT_GRIPPER_Z_LINK",
            "right": "RIGHT_GRIPPER_Z_LINK",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="PELVIS_LINK",
            num_hand_joints=len(ABILITY_HAND_TELEOP_JOINT_ORDER),
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                LocalFrameTaskCfg(
                    frame="LEFT_GRIPPER_Z_LINK",
                    base_link_frame_name="PELVIS_LINK",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                LocalFrameTaskCfg(
                    frame="RIGHT_GRIPPER_Z_LINK",
                    base_link_frame_name="PELVIS_LINK",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                DampingTaskCfg(cost=0.5),
                NullSpacePostureTaskCfg(cost=0.5),
            ],
        ),
    )


@configclass
class AlexAbilityHandObservationsCfg:
    """Observation specifications for Alex with Ability Hands."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "LEFT_GRIPPER_Z_LINK"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "LEFT_GRIPPER_Z_LINK"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "RIGHT_GRIPPER_Z_LINK"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "RIGHT_GRIPPER_Z_LINK"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@register_asset
class AlexAbilityHandEmbodiment(EmbodimentBase):
    """Embodiment for the IHMC Alex V1 robot with Psyonic Ability Hands and PINK IK wrist control.

    Requires alex-models mounted at /models and the ihmc_hands_ros2 package accessible.
    Either set ABILITY_HAND_MODELS_DIR or ensure the SDK layout
    ``<sdk-root>/alex-ros2/ihmc_hands_ros2`` is resolvable from ``_ALEX_MODELS_DIR``'s parent.
    """

    name = "alex_ability_hands"
    default_arm_mode = ArmMode.RIGHT

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        use_tiled_camera: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose)

        merged_urdf = _merge_ability_hands_urdf(ALEX_V1)
        resolved_urdf = _resolve_mesh_paths_ability_hands(merged_urdf, _ALEX_ABILITY_HANDS_RESOLVED_URDF_PATH)

        robot_cfg = copy.deepcopy(ALEX_V1_NUBS_DEFAULT_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.asset_path = resolved_urdf
        # Ability Hand joints have narrow, asymmetric limits — keep full URDF range.
        robot_cfg.soft_joint_pos_limit_factor = 1.0

        # High-stiffness arms + wrists for teleop tracking.
        robot_cfg.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*", ".*WRIST.*", ".*GRIPPER.*"],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=3000.0,
            damping=100.0,
            armature=0.0,
        )
        # Finger joints driven by hand-tracker retargeting (not PINK IK).
        robot_cfg.actuators["hands"] = ImplicitActuatorCfg(
            joint_names_expr=[".*ability_hand.*_q[12]"],
            effort_limit=5.0,
            velocity_limit=10.0,
            stiffness=500.0,
            damping=20.0,
            armature=0.0,
        )

        robot_cfg.init_state.joint_pos = {
            "LEFT_ELBOW_Y": -1.5708,
            "RIGHT_ELBOW_Y": -1.5708,
            **_ABILITY_HAND_DEFAULT_JOINT_POS,
        }

        self.scene_config = AlexSceneCfg()
        self.scene_config.robot = robot_cfg

        self.action_config = AlexAbilityHandActionsCfg()
        # PINK IK: kinematics-only URDF (Alex capsule collisions break urdfdom).
        pink_ik_urdf = _strip_collisions_for_pink_ik(resolved_urdf, _ALEX_ABILITY_HANDS_PINK_IK_URDF_PATH)
        self.action_config.upper_body_ik.controller.urdf_path = pink_ik_urdf
        self.action_config.upper_body_ik.controller.mesh_path = None

        self.observation_config = AlexAbilityHandObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = AlexEventCfg()
        if enable_cameras:
            self.event_config.sync_zed_cameras = EventTerm(
                func=sync_alex_zed_cameras,
                mode="interval",
                interval_range_s=(CONTROL_DT, CONTROL_DT),
            )
            self.event_config.sync_zed_cameras_reset = EventTerm(
                func=sync_alex_zed_cameras,
                mode="reset",
            )
        self.mimic_env = AlexAbilityHandMimicEnv
        self.camera_config = AlexCameraCfg()
        self.camera_config._use_tiled_camera = use_tiled_camera
        self.camera_config.__post_init__()

        self.xr = copy.deepcopy(_ALEX_XR_CFG)


@configclass
class AlexAbilityHandJointPositionActionsCfg:
    """Direct joint-position actions for Alex with Ability Hands.

    Layout matches ``alex_34dof_action_joint_space.yaml``:
      arms+wrists 0-13  (ARM_WRIST_JOINT_NAMES_LIST order)
      hands       14-33 (ABILITY_HAND_TELEOP_JOINT_ORDER)

    ``preserve_order=True`` is required because ABILITY_HAND_TELEOP_JOINT_ORDER
    groups all q1 joints before all q2 joints, while the articulation interleaves
    them per-finger (index_q1, index_q2, middle_q1, …).
    """

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_WRIST_JOINT_NAMES_LIST + ABILITY_HAND_TELEOP_JOINT_ORDER,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
        offset={".*GRIPPER_Z": -math.pi / 2},
    )


@register_asset
class AlexAbilityHandJointPositionEmbodiment(EmbodimentBase):
    """Alex V1 with Ability Hands using direct joint-position actions (no IK).

    Drop-in policy-eval counterpart to :class:`AlexAbilityHandEmbodiment`.
    Accepts 34-DOF absolute joint position commands in the order defined by
    ``alex_34dof_action_joint_space.yaml`` (arms 0-13, hands 14-33).
    Use this embodiment when the policy was trained on direct joint-angle data
    (e.g. ``H2Ozone/alex_demo``) rather than PINK IK wrist-pose targets.
    """

    name = "alex_ability_hands_joint_pos"
    default_arm_mode = ArmMode.RIGHT

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        use_tiled_camera: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose)

        merged_urdf = _merge_ability_hands_urdf(ALEX_V1)
        resolved_urdf = _resolve_mesh_paths_ability_hands(merged_urdf, _ALEX_ABILITY_HANDS_RESOLVED_URDF_PATH)

        robot_cfg = copy.deepcopy(ALEX_V1_NUBS_DEFAULT_CFG)
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.asset_path = resolved_urdf
        robot_cfg.soft_joint_pos_limit_factor = 1.0

        robot_cfg.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*", ".*WRIST.*", ".*GRIPPER.*"],
            effort_limit=torch.inf,
            velocity_limit=torch.inf,
            stiffness=400.0,
            damping=40.0,
            armature=0.0,
        )
        robot_cfg.actuators["hands"] = ImplicitActuatorCfg(
            joint_names_expr=[".*ability_hand.*_q[12]"],
            effort_limit=5.0,
            velocity_limit=10.0,
            stiffness=500.0,
            damping=20.0,
            armature=0.0,
        )

        robot_cfg.init_state.joint_pos = {
            "LEFT_ELBOW_Y": -1.5708,
            "RIGHT_ELBOW_Y": -1.5708,
            "LEFT_GRIPPER_Z": -math.pi / 2,
            "RIGHT_GRIPPER_Z": -math.pi / 2,
            **_ABILITY_HAND_DEFAULT_JOINT_POS,
        }

        self.scene_config = AlexSceneCfg()
        self.scene_config.robot = robot_cfg

        self.action_config = AlexAbilityHandJointPositionActionsCfg()

        self.observation_config = AlexAbilityHandObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = AlexEventCfg()
        if enable_cameras:
            self.event_config.sync_zed_cameras = EventTerm(
                func=sync_alex_zed_cameras,
                mode="interval",
                interval_range_s=(CONTROL_DT, CONTROL_DT),
            )
            self.event_config.sync_zed_cameras_reset = EventTerm(
                func=sync_alex_zed_cameras,
                mode="reset",
            )
        self.camera_config = AlexCameraCfg()
        self.camera_config._use_tiled_camera = use_tiled_camera
        self.camera_config.__post_init__()
