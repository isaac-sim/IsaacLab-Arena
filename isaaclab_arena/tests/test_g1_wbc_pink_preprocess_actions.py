# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for G1 WBC Pink action preprocess_actions (world → robot base frame)."""

import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_END_IDX,
    TORSO_ORIENTATION_RPY_CMD_START_IDX,
)

HEADLESS = True


def _get_g1_pink_env_and_term(simulation_app):
    """Build G1 WBC Pink env at origin with identity orientation; return env and g1_action term."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.g1.g1 import G1WBCPinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    scene = Scene(assets=[background])
    embodiment = G1WBCPinkEmbodiment(enable_cameras=False)
    embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="g1_pink_preprocess_test",
        embodiment=embodiment,
        scene=scene,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()
    term = env.unwrapped.action_manager.get_term("g1_action")
    return env, term


def _test_preprocess_actions_shape(simulation_app) -> bool:
    """preprocess_actions preserves shape (num_envs, action_dim)."""
    env, term = _get_g1_pink_env_and_term(simulation_app)
    try:
        action_dim = term.action_dim
        num_envs = env.unwrapped.num_envs
        actions = torch.zeros(num_envs, action_dim, device=env.unwrapped.device)
        out = term.preprocess_actions(actions)
        assert out.shape == (num_envs, action_dim), f"Expected shape ({num_envs}, {action_dim}), got {out.shape}"
    finally:
        env.close()
    return True


def _test_preprocess_actions_identity_base(simulation_app) -> bool:
    """When robot base has identity quat, wrist in base frame = world pos minus base pos."""
    env, term = _get_g1_pink_env_and_term(simulation_app)
    try:
        device = env.unwrapped.device
        action_dim = term.action_dim
        robot_base_pos = term._asset.data.root_link_pos_w[0, :3]

        # World-frame wrist positions: base + offset (so base-frame offset is known)
        left_offset = torch.tensor([1.0, 2.0, 3.0], device=device)
        right_offset = torch.tensor([4.0, 5.0, 6.0], device=device)
        left_pos_world = robot_base_pos + left_offset
        right_pos_world = robot_base_pos + right_offset

        actions = torch.zeros(1, action_dim, device=device)
        actions[0, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = left_pos_world
        actions[0, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device
        )
        actions[0, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = right_pos_world
        actions[0, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device
        )

        out = term.preprocess_actions(actions)

        # Base frame position = world - base (in world), then rotated by base_inv => offset when base quat is identity
        torch.testing.assert_close(
            out[0, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX], left_offset, atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            out[0, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX], right_offset, atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            out[0, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX],
            actions[0, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX],
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            out[0, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX],
            actions[0, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX],
            atol=1e-5,
            rtol=0,
        )
    finally:
        env.close()
    return True


def _test_preprocess_actions_roundtrip(simulation_app) -> bool:
    """Preprocess world→base; then base→world recovers original (using current robot pose)."""
    import isaaclab.utils.math as math_utils

    env, term = _get_g1_pink_env_and_term(simulation_app)
    try:
        device = env.unwrapped.device
        action_dim = term.action_dim
        asset = term._asset

        robot_base_pos = asset.data.root_link_pos_w[:, :3]
        robot_base_quat = asset.data.root_link_quat_w
        num_envs = robot_base_pos.shape[0]

        # Arbitrary world-frame wrist poses
        left_pos_w = torch.tensor([[1.0, 0.0, 0.5]], device=device).expand(num_envs, 3)
        left_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)
        right_pos_w = torch.tensor([[0.0, 1.0, 0.5]], device=device).expand(num_envs, 3)
        right_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, 4)

        actions = torch.zeros(num_envs, action_dim, device=device)
        actions[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = left_pos_w
        actions[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = left_quat_w
        actions[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = right_pos_w
        actions[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = right_quat_w

        out = term.preprocess_actions(actions)
        left_pos_b = out[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX]
        left_quat_b = out[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX]
        right_pos_b = out[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX]
        right_quat_b = out[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX]

        # Base → world: pos_w = base_pos + quat_apply(base_quat, pos_b), quat_w = quat_mul(base_quat, quat_b)
        left_pos_w_recovered = robot_base_pos + math_utils.quat_apply(robot_base_quat, left_pos_b)
        left_quat_w_recovered = math_utils.quat_mul(robot_base_quat, left_quat_b)
        right_pos_w_recovered = robot_base_pos + math_utils.quat_apply(robot_base_quat, right_pos_b)
        right_quat_w_recovered = math_utils.quat_mul(robot_base_quat, right_quat_b)

        torch.testing.assert_close(left_pos_w_recovered, left_pos_w, atol=1e-5, rtol=0)
        torch.testing.assert_close(left_quat_w_recovered, left_quat_w, atol=1e-5, rtol=0)
        torch.testing.assert_close(right_pos_w_recovered, right_pos_w, atol=1e-5, rtol=0)
        torch.testing.assert_close(right_quat_w_recovered, right_quat_w, atol=1e-5, rtol=0)
    finally:
        env.close()
    return True


def _test_preprocess_actions_does_not_mutate_other_slots(simulation_app) -> bool:
    """Indices outside wrist pos/quat (e.g. hand state, navigate_cmd) are unchanged."""
    env, term = _get_g1_pink_env_and_term(simulation_app)
    try:
        device = env.unwrapped.device
        action_dim = term.action_dim
        actions = torch.zeros(1, action_dim, device=device)
        actions[0, LEFT_HAND_STATE_IDX] = 0.5
        actions[0, RIGHT_HAND_STATE_IDX] = 0.7
        actions[0, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX] = torch.tensor([0.1, 0.2, 0.3], device=device)
        actions[0, BASE_HEIGHT_CMD_START_IDX] = 0.75
        actions[0, TORSO_ORIENTATION_RPY_CMD_START_IDX:TORSO_ORIENTATION_RPY_CMD_END_IDX] = torch.tensor(
            [0.0, 0.0, 0.1], device=device
        )

        out = term.preprocess_actions(actions)

        torch.testing.assert_close(out[0, LEFT_HAND_STATE_IDX], torch.tensor(0.5, device=device), atol=1e-6, rtol=0)
        torch.testing.assert_close(out[0, RIGHT_HAND_STATE_IDX], torch.tensor(0.7, device=device), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            out[0, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX],
            actions[0, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX],
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            out[0, BASE_HEIGHT_CMD_START_IDX], actions[0, BASE_HEIGHT_CMD_START_IDX], atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            out[0, TORSO_ORIENTATION_RPY_CMD_START_IDX:TORSO_ORIENTATION_RPY_CMD_END_IDX],
            actions[0, TORSO_ORIENTATION_RPY_CMD_START_IDX:TORSO_ORIENTATION_RPY_CMD_END_IDX],
            atol=1e-6,
            rtol=0,
        )
    finally:
        env.close()
    return True


def test_g1_wbc_pink_preprocess_actions_shape():
    result = run_simulation_app_function(
        _test_preprocess_actions_shape,
        headless=HEADLESS,
    )
    assert result, "preprocess_actions shape test failed"


def test_g1_wbc_pink_preprocess_actions_identity_base():
    result = run_simulation_app_function(
        _test_preprocess_actions_identity_base,
        headless=HEADLESS,
    )
    assert result, "preprocess_actions identity base test failed"


def test_g1_wbc_pink_preprocess_actions_roundtrip():
    result = run_simulation_app_function(
        _test_preprocess_actions_roundtrip,
        headless=HEADLESS,
    )
    assert result, "preprocess_actions roundtrip test failed"


def test_g1_wbc_pink_preprocess_actions_does_not_mutate_other_slots():
    result = run_simulation_app_function(
        _test_preprocess_actions_does_not_mutate_other_slots,
        headless=HEADLESS,
    )
    assert result, "preprocess_actions other slots test failed"
