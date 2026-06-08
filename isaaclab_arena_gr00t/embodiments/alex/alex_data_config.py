# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
GR00T N1.6 modality config for Alex ability-hands LeRobot datasets.

Matches ``isaaclab_arena_gr00t/embodiments/alex/modality.json`` produced by
``convert_hdf5_to_lerobot.py`` with ``alex_open_microwave_config.yaml``.

Register under ``NEW_EMBODIMENT`` for finetuning::

    --embodiment-tag NEW_EMBODIMENT \\
    --modality-config-path isaaclab_arena_gr00t/embodiments/alex/alex_data_config.py

Action horizon (16) must match the closed-loop policy server config if you
deploy the checkpoint back into Arena.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ALEX_ACTION_HORIZON = 16

_JOINT_STATE_GROUPS = [
    "left_leg",
    "right_leg",
    "spine_neck",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
]

alex_ability_hands_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "zed_left_cam",
            "zed_right_cam",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=_JOINT_STATE_GROUPS,
        sin_cos_embedding_keys=_JOINT_STATE_GROUPS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ALEX_ACTION_HORIZON)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "hands",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(alex_ability_hands_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
