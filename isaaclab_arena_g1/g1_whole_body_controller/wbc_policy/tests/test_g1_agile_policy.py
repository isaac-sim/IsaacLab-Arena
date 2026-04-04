# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the G1AgilePolicy.

These tests run without the full Isaac Lab simulation environment by mocking
the RobotModel and feeding synthetic observations.
"""

import numpy as np
import pathlib
import yaml

import onnxruntime as ort
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WBC_POLICY_DIR = pathlib.Path(__file__).parent.parent
ONNX_MODEL_PATH = WBC_POLICY_DIR / "models" / "agile" / "unitree_g1_velocity_e2e.onnx"
AGILE_CONFIG_PATH = WBC_POLICY_DIR / "config" / "g1_agile.yaml"
WBC_JOINTS_ORDER_PATH = WBC_POLICY_DIR.parent.parent / "g1_env" / "config" / "loco_manip_g1_joints_order_43dof.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MockRobotModel:
    """Minimal mock of RobotModel providing only what G1AgilePolicy needs."""

    def __init__(self):
        with open(WBC_JOINTS_ORDER_PATH) as f:
            self.wbc_g1_joints_order = yaml.safe_load(f)

        # Build a reverse mapping: index -> name
        self._idx_to_name = {v: k for k, v in self.wbc_g1_joints_order.items()}

        # Lower body = left_leg(0-5) + right_leg(6-11) + waist(12-14)
        self._lower_body_indices = list(range(15))

    def get_joint_group_indices(self, group_name):
        if group_name == "lower_body":
            return self._lower_body_indices
        raise ValueError(f"MockRobotModel: unsupported group '{group_name}'")


def make_observation(num_envs: int = 1, num_joints: int = 43) -> dict:
    """Create a synthetic observation dict matching prepare_observations() output."""
    return {
        "q": np.zeros((num_envs, num_joints), dtype=np.float32),
        "dq": np.zeros((num_envs, num_joints), dtype=np.float32),
        "floating_base_pose": np.tile(
            np.array([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (num_envs, 1),
        ),  # pos + quat (w,x,y,z)
        "floating_base_vel": np.zeros((num_envs, 6), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestOnnxModelDirect:
    """Test the ONNX model directly (no policy wrapper)."""

    def test_model_loads(self):
        session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        assert len(session.get_inputs()) == 12
        assert len(session.get_outputs()) == 10

    def test_model_input_names(self):
        session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        input_names = {inp.name for inp in session.get_inputs()}
        expected = {
            "root_link_quat_w",
            "root_ang_vel_b",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "last_actions",
            "base_ang_vel_history",
            "projected_gravity_history",
            "velocity_commands_history",
            "controlled_joint_pos_history",
            "controlled_joint_vel_history",
            "actions_history",
        }
        assert input_names == expected

    def test_model_output_names(self):
        session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        output_names = {out.name for out in session.get_outputs()}
        expected = {
            "action_joint_pos",
            "action_joint_pos_kp_gains",
            "action_joint_pos_kd_gains",
            "last_actions_out",
            "base_ang_vel_history_out",
            "projected_gravity_history_out",
            "velocity_commands_history_out",
            "controlled_joint_pos_history_out",
            "controlled_joint_vel_history_out",
            "actions_history_out",
        }
        assert output_names == expected

    def test_model_inference_with_zeros(self):
        """Run the model with all-zero inputs and verify output shapes."""
        session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        inputs = {
            "root_link_quat_w": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "root_ang_vel_b": np.zeros((1, 3), dtype=np.float32),
            "velocity_commands": np.zeros((1, 3), dtype=np.float32),
            "joint_pos": np.zeros((1, 29), dtype=np.float32),
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "last_actions": np.zeros((1, 14), dtype=np.float32),
            "base_ang_vel_history": np.zeros((1, 5, 3), dtype=np.float32),
            "projected_gravity_history": np.zeros((1, 5, 3), dtype=np.float32),
            "velocity_commands_history": np.zeros((1, 5, 3), dtype=np.float32),
            "controlled_joint_pos_history": np.zeros((1, 5, 14), dtype=np.float32),
            "controlled_joint_vel_history": np.zeros((1, 5, 14), dtype=np.float32),
            "actions_history": np.zeros((1, 5, 14), dtype=np.float32),
        }

        output_names = [out.name for out in session.get_outputs()]
        outputs = session.run(output_names, inputs)
        result = dict(zip(output_names, outputs))

        assert result["action_joint_pos"].shape == (1, 14)
        assert result["action_joint_pos_kp_gains"].shape == (1, 14)
        assert result["action_joint_pos_kd_gains"].shape == (1, 14)
        assert result["last_actions_out"].shape == (1, 14)
        assert result["base_ang_vel_history_out"].shape == (1, 5, 3)
        assert result["actions_history_out"].shape == (1, 5, 14)

        # Actions should be finite
        assert np.all(np.isfinite(result["action_joint_pos"]))
        # Gains should be positive
        assert np.all(result["action_joint_pos_kp_gains"] > 0)
        assert np.all(result["action_joint_pos_kd_gains"] > 0)


class TestJointMappings:
    """Test that the joint ordering mappings are correct."""

    def test_agile_config_loads(self):
        with open(AGILE_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        assert len(config["onnx_input_joint_names"]) == 29
        assert len(config["controlled_joint_names"]) == 14

    def test_wbc_to_agile_input_mapping(self):
        """Verify the input mapping selects correct joints from WBC order."""
        with open(WBC_JOINTS_ORDER_PATH) as f:
            wbc_order = yaml.safe_load(f)
        with open(AGILE_CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        mapping = [wbc_order[name] for name in config["onnx_input_joint_names"]]
        # All indices should be valid (0-42)
        assert all(0 <= idx <= 42 for idx in mapping)
        # Should have 29 unique indices (one per body joint)
        assert len(set(mapping)) == 29
        # First entry is left_hip_pitch at WBC index 0
        assert mapping[0] == 0
        # Second entry is right_hip_pitch at WBC index 6
        assert mapping[1] == 6

    def test_agile_output_to_lower_body_mapping(self):
        """Verify the output mapping covers all lower body joints except waist_yaw."""
        with open(WBC_JOINTS_ORDER_PATH) as f:
            wbc_order = yaml.safe_load(f)
        with open(AGILE_CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        lower_body_indices = list(range(15))
        output_mapping = []
        for name in config["controlled_joint_names"]:
            wbc_idx = wbc_order[name]
            lb_pos = lower_body_indices.index(wbc_idx)
            output_mapping.append(lb_pos)

        # Should cover 14 of the 15 lower body positions
        assert len(set(output_mapping)) == 14
        # waist_yaw is at lower_body position 12 and should NOT be in the mapping
        assert 12 not in output_mapping  # waist_yaw_joint = WBC index 12 = lb pos 12


class TestG1AgilePolicy:
    """Test the full G1AgilePolicy class."""

    @pytest.fixture
    def policy(self):
        robot_model = MockRobotModel()
        return _create_policy(robot_model, num_envs=1)

    @pytest.fixture
    def policy_multi_env(self):
        robot_model = MockRobotModel()
        return _create_policy(robot_model, num_envs=3)

    def test_init(self, policy):
        assert policy.num_envs == 1
        assert policy.num_lower_body == 15
        assert len(policy.wbc_to_agile_input) == 29
        assert len(policy.agile_output_to_lower_body) == 14

    def test_get_action_shape(self, policy):
        obs = make_observation(num_envs=1)
        policy.set_observation(obs)
        action = policy.get_action()
        assert "body_action" in action
        assert action["body_action"].shape == (1, 15)

    def test_get_action_finite(self, policy):
        obs = make_observation(num_envs=1)
        policy.set_observation(obs)
        action = policy.get_action()
        assert np.all(np.isfinite(action["body_action"]))

    def test_get_action_waist_yaw_zero(self, policy):
        """waist_yaw (lower_body position 12) should always be 0."""
        obs = make_observation(num_envs=1)
        policy.set_observation(obs)
        action = policy.get_action()
        assert action["body_action"][0, 12] == 0.0

    def test_get_action_multi_env(self, policy_multi_env):
        obs = make_observation(num_envs=3)
        policy_multi_env.set_observation(obs)
        action = policy_multi_env.get_action()
        assert action["body_action"].shape == (3, 15)
        assert np.all(np.isfinite(action["body_action"]))

    def test_set_goal_navigate_cmd(self, policy):
        cmd = np.array([[0.5, 0.0, 0.1]], dtype=np.float32)
        policy.set_goal({"navigate_cmd": cmd})
        np.testing.assert_array_equal(policy.cmd, cmd)

    def test_reset(self, policy):
        """After reset, state should be zeroed and action still valid."""
        obs = make_observation(num_envs=1)
        policy.set_observation(obs)
        policy.get_action()  # populate state

        import torch

        policy.reset(torch.tensor([0]))
        policy.set_observation(obs)
        action = policy.get_action()
        assert np.all(np.isfinite(action["body_action"]))

    def test_multiple_steps(self, policy):
        """Run multiple steps to verify feedback state propagation."""
        obs = make_observation(num_envs=1)
        policy.set_goal({"navigate_cmd": np.array([[0.3, 0.0, 0.0]], dtype=np.float32)})

        for _ in range(5):
            policy.set_observation(obs)
            action = policy.get_action()
            assert np.all(np.isfinite(action["body_action"]))

        # After several steps with non-zero command, state should be non-trivial
        state = policy.states[0]
        assert not np.allclose(state["last_actions"], 0.0)

    def test_no_observation_raises(self, policy):
        with pytest.raises(ValueError, match="No observation set"):
            policy.get_action()

    def test_toggle_policy_action(self, policy):
        """Toggling use_policy_action should switch between NN output and passthrough."""
        obs = make_observation(num_envs=1)
        # Set non-zero joint positions so passthrough differs from NN output
        obs["q"][0, :15] = 0.1
        policy.set_observation(obs)
        action_nn = policy.get_action()["body_action"].copy()

        # Toggle off
        policy.set_goal({"toggle_policy_action": True})
        assert not policy.use_policy_action

        policy.set_observation(obs)
        action_passthrough = policy.get_action()["body_action"].copy()

        # Passthrough should return observed lower body joint positions
        np.testing.assert_array_almost_equal(action_passthrough[0, :15], obs["q"][0, :15])

        # NN output should differ from passthrough (unless extremely unlikely coincidence)
        assert not np.allclose(action_nn, action_passthrough)


class TestAgileStability:
    """Multi-step stability tests verifying the policy does not diverge.

    These are unit-level proxies for the full simulation integration test. They
    run many policy steps with standing-upright observations and verify that the
    outputs remain bounded and physically reasonable (i.e., the policy would keep
    the G1 robot balanced in simulation with root z > 0.5 m).
    """

    @pytest.fixture
    def policy(self):
        robot_model = MockRobotModel()
        return _create_policy(robot_model, num_envs=1)

    def test_standing_stability_100_steps(self, policy):
        """Run 100 steps with zero velocity command; outputs should stay bounded."""
        obs = make_observation(num_envs=1)
        # Standing upright: z=0.75, identity quaternion
        max_joint_pos = 0.0
        for _ in range(100):
            policy.set_observation(obs)
            action = policy.get_action()
            body_action = action["body_action"]
            assert np.all(np.isfinite(body_action)), "Policy produced non-finite output"
            max_joint_pos = max(max_joint_pos, np.max(np.abs(body_action)))

        # Joint positions should remain within physically reasonable bounds (< 3 rad)
        assert max_joint_pos < 3.0, f"Joint positions diverged: max |pos| = {max_joint_pos:.3f} rad"

    def test_walking_stability_100_steps(self, policy):
        """Run 100 steps with forward velocity command; outputs should stay bounded."""
        obs = make_observation(num_envs=1)
        policy.set_goal({"navigate_cmd": np.array([[0.5, 0.0, 0.0]], dtype=np.float32)})

        for step in range(100):
            policy.set_observation(obs)
            action = policy.get_action()
            body_action = action["body_action"]
            assert np.all(np.isfinite(body_action)), f"Policy produced non-finite output at step {step}"
            # Each joint target should be within a reasonable range
            assert (
                np.max(np.abs(body_action)) < 3.0
            ), f"Joint positions diverged at step {step}: max |pos| = {np.max(np.abs(body_action)):.3f} rad"

    def test_root_position_above_half_meter_proxy(self, policy):
        """Verify that the policy produces actions consistent with maintaining balance.

        This test verifies that the AGILE policy produces bounded, non-divergent
        joint targets over many steps -- a necessary condition for the robot to
        maintain its root position above 0.5 m in simulation.

        For the full simulation-based verification (root z > 0.5 m with Isaac Sim
        physics), run the integration test in the Docker container:
            /isaac-sim/python.sh -m pytest -sv -k test_g1_agile_root_z_above_half_meter
        """
        obs = make_observation(num_envs=1)
        # Start at z=0.75m (typical G1 standing height)
        initial_z = obs["floating_base_pose"][0, 2]
        assert initial_z == 0.75

        actions_over_time = []
        for _ in range(200):
            policy.set_observation(obs)
            action = policy.get_action()
            body_action = action["body_action"]
            actions_over_time.append(body_action.copy())
            assert np.all(np.isfinite(body_action))

        # Verify no single action is extreme (which would cause the robot to fall)
        all_actions = np.concatenate(actions_over_time, axis=0)
        assert np.max(np.abs(all_actions)) < 3.0, "Policy actions too extreme -- robot would likely fall"

        # Verify actions are not all zero (policy is actively balancing)
        assert not np.allclose(
            all_actions, 0.0, atol=1e-4
        ), "Policy outputs are all near-zero -- not actively balancing"


# ---------------------------------------------------------------------------
# Helper to create policy (avoids importing isaaclab_arena_g1 module)
# ---------------------------------------------------------------------------
def _create_policy(robot_model, num_envs):
    """Create a G1AgilePolicy without going through the module import."""
    import sys

    # Add the parent packages to sys.path so we can import directly
    arena_root = WBC_POLICY_DIR.parent.parent.parent
    if str(arena_root) not in sys.path:
        sys.path.insert(0, str(arena_root))

    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.g1_agile_policy import G1AgilePolicy

    return G1AgilePolicy(
        robot_model=robot_model,
        config_path="config/g1_agile.yaml",
        model_path="models/agile/unitree_g1_velocity_e2e.onnx",
        num_envs=num_envs,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
