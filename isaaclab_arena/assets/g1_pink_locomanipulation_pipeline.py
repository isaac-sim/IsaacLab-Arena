# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom IsaacTeleop pipeline for G1 WBC Pink: 23D output (20D + 3 torso zeros).

What each retargeter.connect() does
------------------------------------
- **left_se3.connect({LEFT: transformed_controllers.output(LEFT)})**
  Binds the left controller's *transformed* grip pose (position + quaternion in
  world frame) as the sole input to the left SE3 retargeter. The retargeter
  outputs a 7D ee_pose (position + quat) with configurable rotation offsets.

- **right_se3.connect({RIGHT: transformed_controllers.output(RIGHT)})**
  Same for the right controller → right SE3 retargeter → 7D ee_pose.

- **left_trihand.connect({LEFT: transformed_controllers.output(LEFT)})**
  Feeds the left controller (buttons, trigger, squeeze) into the left
  TriHand retargeter. It outputs 7 hand joint scalars (thumb/index/middle)
  derived from trigger and squeeze.

- **right_trihand.connect({RIGHT: transformed_controllers.output(RIGHT)})**
  Same for the right controller → right TriHand → 7 hand joint scalars.

- **locomotion.connect({controller_left: controllers.output(LEFT), controller_right: ...})**
  Feeds *raw* (untransformed) left and right controller data into the
  locomotion retargeter. It uses thumbsticks to produce a 4D root command
  [vel_x, vel_y, rot_vel_z, hip_height]. Raw controllers are used so
  thumbstick values are in controller space.

- **reorderer.connect({...})**
  Connects each retargeter's output to the TensorReorderer's named inputs.
  Adds three 0s for the torso.
  The reorderer flattens and reorders them into a single 23D action vector.
"""

def _build_g1_pink_locomanipulation_pipeline():
    """Build an IsaacTeleop retargeting pipeline for G1 WBC Pink locomanipulation.

    Same sources as Isaac Lab's G1 locomanipulation (Se3 wrists, TriHand hands,
    Locomotion root), plus a ValueInput for torso_rpy (zeros). Output is 23D:
    [left_gripper(1), right_gripper(1), left_wrist(7), right_wrist(7), locomotion(4), torso_rpy(3)].

    Returns:
        OutputCombiner with a single "action" output (23D flattened tensor).
    """
    from isaacteleop.retargeters import (
        LocomotionRootCmdRetargeter,
        LocomotionRootCmdRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
        TriHandMotionControllerConfig,
        TriHandMotionControllerRetargeter,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # connect(): binds transformed left/right controller grip pose -> 7D ee_pose each.
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=45.0,
        target_offset_pitch=180.0,
        target_offset_yaw=-90.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=-135.0,
        target_offset_pitch=0.0,
        target_offset_yaw=90.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # TriHand Motion Controller Retargeters (for gripper scalar per hand)
    # -------------------------------------------------------------------------
    # connect(): binds transformed left/right controller -> 7 hand joint scalars each.
    hand_joint_names = [
        "thumb_rotation",
        "thumb_proximal",
        "thumb_distal",
        "index_proximal",
        "index_distal",
        "middle_proximal",
        "middle_distal",
    ]
    left_trihand_cfg = TriHandMotionControllerConfig(
        hand_joint_names=hand_joint_names,
        controller_side="left",
    )
    left_trihand = TriHandMotionControllerRetargeter(left_trihand_cfg, name="trihand_left")
    connected_left_trihand = left_trihand.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_trihand_cfg = TriHandMotionControllerConfig(
        hand_joint_names=hand_joint_names,
        controller_side="right",
    )
    right_trihand = TriHandMotionControllerRetargeter(right_trihand_cfg, name="trihand_right")
    connected_right_trihand = right_trihand.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # Locomotion Root Command Retargeter
    # -------------------------------------------------------------------------
    # connect(): binds raw left/right controller (thumbsticks) -> 4D root_command.
    locomotion_cfg = LocomotionRootCmdRetargeterConfig(
        initial_hip_height=0.72,
        movement_scale=0.5,
        rotation_scale=0.35,
        dt=1.0 / 100.0,
    )
    locomotion = LocomotionRootCmdRetargeter(locomotion_cfg, name="locomotion")
    connected_locomotion = locomotion.connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: 23D for G1 WBC Pink [20D above + torso_rpy(3) from ConstantRetargeter]
    # -------------------------------------------------------------------------
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_hand_elements = [
        "l_thumb_rotation",
        "l_thumb_proximal",
        "l_thumb_distal",
        "l_index_proximal",
        "l_index_distal",
        "l_middle_proximal",
        "l_middle_distal",
    ]
    right_hand_elements = [
        "r_thumb_rotation",
        "r_thumb_proximal",
        "r_thumb_distal",
        "r_index_proximal",
        "r_index_distal",
        "r_middle_proximal",
        "r_middle_distal",
    ]
    locomotion_elements = ["loco_vel_x", "loco_vel_y", "loco_rot_vel_z", "loco_hip_height"]
    torso_elements = ["torso_x", "torso_y", "torso_z"]

    output_order = (
        ["l_thumb_rotation", "r_thumb_rotation"]  # left_gripper(1), right_gripper(1)
        + left_ee_elements
        + right_ee_elements
        + locomotion_elements
        + torso_elements  # 3 zeros
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_elements,
            "right_hand_joints": right_hand_elements,
            "locomotion": locomotion_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
            "locomotion": "array",
        },
    )
    # connect(): binds each retargeter output to the reorderer; flattens to 23D action.
    # torso_rpy comes from ConstantRetargeter(output_dims=3, value=0.0).
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_trihand.output("hand_joints"),
            "right_hand_joints": connected_right_trihand.output("hand_joints"),
            "locomotion": connected_locomotion.output("root_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})
