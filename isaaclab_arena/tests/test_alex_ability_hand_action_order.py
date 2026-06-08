# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Alex Ability Hands teleop ↔ Pink IK action tensor layout."""


def _simulate_tensor_reorderer_mapping(input_config: dict[str, list[str]], output_order: list[str]) -> list[tuple[str, int, int]]:
    """Replicate TensorReorderer index mapping (see isaacteleop.retargeters.tensor_reorderer)."""
    mapping: list[tuple[str, int, int]] = []
    for target_idx, element_name in enumerate(output_order):
        for input_name, source_elements in input_config.items():
            if element_name in source_elements:
                source_idx = source_elements.index(element_name)
                mapping.append((input_name, source_idx, target_idx))
                break
    return mapping


def test_ability_hand_action_dimensions():
    from isaaclab_arena.embodiments.alex.alex import (
        ALEX_ABILITY_HAND_HAND_ACTION_DIM,
        ALEX_ABILITY_HAND_TOTAL_ACTION_DIM,
        ALEX_ABILITY_HAND_WRIST_ACTION_DIM,
    )

    assert ALEX_ABILITY_HAND_WRIST_ACTION_DIM == 14
    assert ALEX_ABILITY_HAND_HAND_ACTION_DIM == 20
    assert ALEX_ABILITY_HAND_TOTAL_ACTION_DIM == 34


def test_pink_cfg_hand_joint_order_matches_teleop():
    from isaaclab_arena.embodiments.alex.alex import (
        ABILITY_HAND_TELEOP_JOINT_ORDER,
        AlexAbilityHandActionsCfg,
    )

    cfg = AlexAbilityHandActionsCfg()
    assert cfg.upper_body_ik.hand_joint_names == ABILITY_HAND_TELEOP_JOINT_ORDER
    assert cfg.upper_body_ik.controller.num_hand_joints == len(ABILITY_HAND_TELEOP_JOINT_ORDER)


def test_teleop_output_order_matches_pink_slices():
    from isaaclab_arena.embodiments.alex.alex import (
        ALEX_ABILITY_HAND_HAND_ACTION_DIM,
        ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS,
        ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS,
        ALEX_ABILITY_HAND_WRIST_ACTION_DIM,
        ABILITY_HAND_TELEOP_JOINT_ORDER,
        build_alex_ability_hand_teleop_action_order,
    )

    action_order = build_alex_ability_hand_teleop_action_order()
    assert len(action_order) == ALEX_ABILITY_HAND_WRIST_ACTION_DIM + ALEX_ABILITY_HAND_HAND_ACTION_DIM

    # PinkInverseKinematicsAction: frame tasks first, then actions[:, -num_hand_joints:].
    assert action_order[:7] == ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS
    assert action_order[7:14] == ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS
    assert action_order[-ALEX_ABILITY_HAND_HAND_ACTION_DIM:] == ABILITY_HAND_TELEOP_JOINT_ORDER

    # LocalFrameTask order in AlexAbilityHandActionsCfg: left wrist, then right wrist.
    assert action_order[0] == "l_pos_x"
    assert action_order[7] == "r_pos_x"


def test_tensor_reorderer_maps_all_hand_joints():
    from isaaclab_arena.embodiments.alex.alex import (
        ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS,
        ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS,
        ABILITY_HAND_TELEOP_JOINT_ORDER,
        ability_hand_full_joint_names,
        build_alex_ability_hand_teleop_action_order,
    )

    left_full = ability_hand_full_joint_names("left")
    right_full = ability_hand_full_joint_names("right")
    input_config = {
        "left_ee_pose": ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS,
        "right_ee_pose": ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS,
        "left_hand_joints": left_full,
        "right_hand_joints": right_full,
    }
    output_order = build_alex_ability_hand_teleop_action_order()
    mapping = _simulate_tensor_reorderer_mapping(input_config, output_order)

    mapped_targets = {target_idx for _, _, target_idx in mapping}
    hand_start = len(ALEX_ABILITY_HAND_LEFT_EE_ACTION_KEYS) + len(ALEX_ABILITY_HAND_RIGHT_EE_ACTION_KEYS)
    hand_targets = set(range(hand_start, hand_start + len(ABILITY_HAND_TELEOP_JOINT_ORDER)))

    assert hand_targets.issubset(mapped_targets), (
        "TensorReorderer must map every hand joint name into the teleop action tensor. "
        f"Missing target indices: {sorted(hand_targets - mapped_targets)}"
    )

    for joint_name in ABILITY_HAND_TELEOP_JOINT_ORDER:
        target_idx = hand_start + ABILITY_HAND_TELEOP_JOINT_ORDER.index(joint_name)
        matches = [m for m in mapping if m[2] == target_idx]
        assert len(matches) == 1, f"Expected exactly one mapping for {joint_name}, got {matches}"
        input_name, source_idx, _ = matches[0]
        if joint_name.startswith("left_"):
            assert input_name == "left_hand_joints"
            assert left_full[source_idx] == joint_name
        else:
            assert input_name == "right_hand_joints"
            assert right_full[source_idx] == joint_name


def test_mimic_gripper_pack_unpack_roundtrip():
    import torch

    from isaaclab_arena.embodiments.alex.alex import (
        ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM,
        ALEX_ABILITY_HAND_TOTAL_ACTION_DIM,
        ALEX_ABILITY_HAND_WRIST_ACTION_DIM,
        _pack_ability_hand_teleop_block,
        _unpack_ability_hand_gripper_actions,
    )

    left = torch.arange(ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM, dtype=torch.float32)
    right = torch.arange(10, 10 + ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM, dtype=torch.float32)
    packed = _pack_ability_hand_teleop_block(left, right)
    assert packed.shape == (20,)

    actions = torch.zeros(ALEX_ABILITY_HAND_TOTAL_ACTION_DIM, dtype=torch.float32)
    actions[ALEX_ABILITY_HAND_WRIST_ACTION_DIM :] = packed
    unpacked = _unpack_ability_hand_gripper_actions(actions)
    assert torch.allclose(unpacked["left"], left)
    assert torch.allclose(unpacked["right"], right)


def test_ability_hand_mimic_env_action_dim():
    import torch

    from isaaclab_arena.embodiments.alex.alex import (
        ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM,
        ALEX_ABILITY_HAND_TOTAL_ACTION_DIM,
        AlexAbilityHandMimicEnv,
    )

    env = object.__new__(AlexAbilityHandMimicEnv)
    left_pose = torch.eye(4)
    right_pose = torch.eye(4)
    left_hand = torch.zeros(ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM)
    right_hand = torch.zeros(ALEX_ABILITY_HAND_PER_HAND_GRIPPER_DIM)
    action = env.target_eef_pose_to_action(
        {"left": left_pose, "right": right_pose},
        {"left": left_hand, "right": right_hand},
    )
    assert action.shape == (ALEX_ABILITY_HAND_TOTAL_ACTION_DIM,)


def test_hand_joint_order_is_permutation_of_urdf_joint_list():
    from isaaclab_arena.embodiments.alex.alex import (
        ABILITY_HAND_JOINT_NAMES_LIST,
        ABILITY_HAND_TELEOP_JOINT_ORDER,
    )

    assert sorted(ABILITY_HAND_TELEOP_JOINT_ORDER) == sorted(ABILITY_HAND_JOINT_NAMES_LIST)
    assert len(ABILITY_HAND_TELEOP_JOINT_ORDER) == len(set(ABILITY_HAND_TELEOP_JOINT_ORDER))
