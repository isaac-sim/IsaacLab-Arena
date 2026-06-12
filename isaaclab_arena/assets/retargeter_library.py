# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC, abstractmethod
from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter

_ABILITY_HAND_DEX_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "embodiments",
    "alex",
    "data",
    "configs",
    "dex-retargeting",
)


class RetargetterBase(ABC):
    """Base class for teleop retargeter entries in the Arena registry.

    Subclasses associate a (device, embodiment) pair with a pipeline builder
    function compatible with ``IsaacTeleopCfg.pipeline_builder``.
    """

    device: str
    embodiment: str

    @abstractmethod
    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        """Return an isaacteleop pipeline builder callable, or None if not applicable."""
        raise NotImplementedError


@register_retargeter
class AlexPinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for Alex with PINK IK arm control (no fingers)."""

    device = "openxr"
    embodiment = "alex_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        return _build_alex_pipeline


def _build_alex_pipeline():
    """Build an IsaacTeleop retargeting pipeline for Alex arm teleoperation.

    Creates two Se3AbsRetargeters for left and right elbow-tip pose tracking.
    Output is a 14-D tensor: [left_pose(7), right_pose(7)].
    """
    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=0.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

    # Right arm: 180° yaw offset — Alex's right EEF frame is 180° rotated from the OpenXR frame.
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=180.0,
        ),
        name="right_ee_pose",
    )
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
        },
        output_order=left_ee_elements + right_ee_elements,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
        },
    )
    connected_reorderer = reorderer.connect({
        "left_ee_pose": connected_left_se3.output("ee_pose"),
        "right_ee_pose": connected_right_se3.output("ee_pose"),
    })
    return OutputCombiner({"action": connected_reorderer.output("output")})


def _build_alex_ability_hands_pipeline(robot_version: str = "V1"):
    """Build IsaacTeleop pipeline for Alex with PINK IK wrists and dex hand retargeting.

    Output tensor layout: [left_wrist(7), right_wrist(7), hand_joints(20)].
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from isaaclab_arena.assets.ability_hand_joint_expand_retargeter import AbilityHandJointExpandRetargeter
    from isaaclab_arena.embodiments.alex.alex import (
        ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS,
        ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS,
        ability_hand_full_joint_names,
        ability_hand_independent_joint_names,
        build_alex_ability_hand_teleop_action_order,
        _resolve_standalone_hand_urdf,
    )

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=0.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=180.0,
        ),
        name="right_ee_pose",
    )
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)
    left_independent_joint_names = ability_hand_independent_joint_names("left")
    right_independent_joint_names = ability_hand_independent_joint_names("right")
    left_full_joint_names = ability_hand_full_joint_names("left")
    right_full_joint_names = ability_hand_full_joint_names("right")

    left_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=os.path.join(_ABILITY_HAND_DEX_CONFIG_DIR, "ability_hand_left_dexpilot.yml"),
            hand_urdf=_resolve_standalone_hand_urdf("left", robot_version),
            hand_joint_names=left_independent_joint_names,
            hand_side="left",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="left_hand_dex",
    )
    connected_left_dex = left_dex.connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})

    right_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=os.path.join(_ABILITY_HAND_DEX_CONFIG_DIR, "ability_hand_right_dexpilot.yml"),
            hand_urdf=_resolve_standalone_hand_urdf("right", robot_version),
            hand_joint_names=right_independent_joint_names,
            hand_side="right",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="right_hand_dex",
    )
    connected_right_dex = right_dex.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    left_expand = AbilityHandJointExpandRetargeter(
        left_independent_joint_names,
        left_full_joint_names,
        name="left_hand_expand",
    )
    connected_left_expand = left_expand.connect({
        "hand_joints": connected_left_dex.output("hand_joints"),
    })

    right_expand = AbilityHandJointExpandRetargeter(
        right_independent_joint_names,
        right_full_joint_names,
        name="right_hand_expand",
    )
    connected_right_expand = right_expand.connect({
        "hand_joints": connected_right_dex.output("hand_joints"),
    })

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS,
            "right_ee_pose": ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS,
            "left_hand_joints": left_full_joint_names,
            "right_hand_joints": right_full_joint_names,
        },
        output_order=build_alex_ability_hand_teleop_action_order(),
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect({
        "left_ee_pose": connected_left_se3.output("ee_pose"),
        "right_ee_pose": connected_right_se3.output("ee_pose"),
        "left_hand_joints": connected_left_expand.output("hand_joints"),
        "right_hand_joints": connected_right_expand.output("hand_joints"),
    })
    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_dex, right_dex]


@register_retargeter
class AlexAbilityHandIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for Alex with Psyonic Ability Hands and PINK IK wrist control."""

    device = "openxr"
    embodiment = "alex_ability_hands"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        return lambda: _build_alex_ability_hands_pipeline()[0]

    def get_retargeters_to_tune(self, embodiment: object) -> Callable:
        return lambda: _build_alex_ability_hands_pipeline()[1]


@register_retargeter
class AlexV2PinkIsaacTeleopRetargeter(AlexPinkIsaacTeleopRetargeter):
    """Isaac Teleop pipeline builder for Alex V2 with PINK IK arm control (no fingers)."""

    embodiment = "alex_v2_pink"


@register_retargeter
class AlexV2AbilityHandIsaacTeleopRetargeter(AlexAbilityHandIsaacTeleopRetargeter):
    """Isaac Teleop pipeline builder for Alex V2 with Psyonic Ability Hands and PINK IK wrist control."""

    embodiment = "alex_v2_ability_hands"

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from isaaclab_arena.embodiments.alex.alex import ALEX_V2

        return lambda: _build_alex_ability_hands_pipeline(ALEX_V2)[0]

    def get_retargeters_to_tune(self, embodiment: object) -> Callable:
        from isaaclab_arena.embodiments.alex.alex import ALEX_V2

        return lambda: _build_alex_ability_hands_pipeline(ALEX_V2)[1]


@register_retargeter
class GR1T2PinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for GR1T2 with Pink IK and dex hand retargeting."""

    device = "openxr"
    embodiment = "gr1_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
            _build_gr1t2_pickplace_pipeline,
        )

        return lambda: _build_gr1t2_pickplace_pipeline()[0]


@register_retargeter
class G1WbcPinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for G1 WBC Pink (locomanipulation: wrist + TriHand + locomotion)."""

    device = "openxr"
    embodiment = "g1_wbc_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from isaaclab_arena_g1.teleop.g1_pink_locomanipulation_pipeline import _build_g1_pink_locomanipulation_pipeline

        return _build_g1_pink_locomanipulation_pipeline


@register_retargeter
class G1WbcAgilePinkIsaacTeleopRetargeter(RetargetterBase):
    """Isaac Teleop pipeline builder for G1 WBC AGILE with Pink IK."""

    device = "openxr"
    embodiment = "g1_wbc_agile_pink"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from isaaclab_arena_g1.teleop.g1_pink_locomanipulation_pipeline import _build_g1_pink_locomanipulation_pipeline

        return _build_g1_pink_locomanipulation_pipeline


@register_retargeter
class FrankaKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "franka_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class FrankaSpaceMouseRetargeter(RetargetterBase):
    device = "spacemouse"
    embodiment = "franka_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class DroidDifferentialIKKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "droid_differential_ik"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class AgibotKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "agibot"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class GalbotKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "galbot"

    def __init__(self):
        pass

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
