# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OpenXR motion-controller pipeline for a single IK arm and binary gripper."""


def build_single_arm_openxr_pipeline():
    """Map the right XR controller pose and trigger to a 7D relative-IK action."""
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    ee_delta = Se3RelRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            delta_pos_scale_factor=10.0,
            delta_rot_scale_factor=10.0,
            alpha_pos=0.5,
            alpha_rot=0.5,
        ),
        name="right_controller_ee_delta",
    ).connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})

    gripper = GripperRetargeter(
        GripperRetargeterConfig(hand_side="right", controller_threshold=0.5),
        name="right_controller_gripper",
    ).connect({
        HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
    })

    pose_elements = ["dx", "dy", "dz", "rx", "ry", "rz"]
    gripper_elements = ["gripper"]
    action = TensorReorderer(
        input_config={
            "ee_delta": pose_elements,
            "gripper": gripper_elements,
        },
        output_order=pose_elements + gripper_elements,
        input_types={
            "ee_delta": "array",
            "gripper": "scalar",
        },
        name="single_arm_action",
    ).connect({
        "ee_delta": ee_delta.output("ee_delta"),
        "gripper": gripper.output("gripper_command"),
    })

    return OutputCombiner({"action": action.output("output")})
