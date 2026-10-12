# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted EEF goal commands with calibrated observations and measured completion feedback."""

import json
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy import (
    AgentActionAdapter,
    VLMAgentPolicy,
    VLMAgentPolicyCfg,
    VLMObservationAdapter,
    encode_image,
)


def pose_error(target, measured):
    """Return translation distance and shortest quaternion rotation error for XYZ/XYZW poses."""
    dot = np.dot(target[3:7], measured[3:7]) / (np.linalg.norm(target[3:7]) * np.linalg.norm(measured[3:7]))
    return (
        float(np.linalg.norm(target[:3] - measured[:3])),
        float(2 * np.arccos(np.clip(abs(dot), 0, 1))),
    )


@dataclass
class AgentCommandedGoalPolicyCfg(VLMAgentPolicyCfg):
    """Configure calibrated goal decisions and retained textual history."""

    decision_history: int = 16
    trace_actions: bool = True
    min_convergence_steps: int = 6
    """Minimum executed env steps before a move may be marked converged."""

    position_convergence_m: float = 0.005
    """Maximum measured translation error to treat a move as converged, in meters."""

    rotation_convergence_rad: float = 0.05
    """Maximum measured rotation error to treat a move as converged, in radians."""

    reference_position_tolerance_m: float = 1e-6
    """Maximum remaining IK-reference translation to treat the reference as settled, in meters."""

    reference_rotation_tolerance_rad: float = 1e-6
    """Maximum remaining IK-reference rotation to treat the reference as settled, in radians."""

    def __post_init__(self):
        super().__post_init__()
        assert self.min_convergence_steps >= 1, f"min_convergence_steps must be >= 1, got {self.min_convergence_steps}"
        assert self.position_convergence_m > 0, f"position_convergence_m must be > 0, got {self.position_convergence_m}"
        assert (
            self.rotation_convergence_rad > 0
        ), f"rotation_convergence_rad must be > 0, got {self.rotation_convergence_rad}"
        assert (
            self.reference_position_tolerance_m >= 0
        ), f"reference_position_tolerance_m must be >= 0, got {self.reference_position_tolerance_m}"
        assert (
            self.reference_rotation_tolerance_rad >= 0
        ), f"reference_rotation_tolerance_rad must be >= 0, got {self.reference_rotation_tolerance_rad}"


@register_policy
class AgentCommandedGoalPolicy(VLMAgentPolicy, PolicyBase[AgentCommandedGoalPolicyCfg]):
    """Execute EEF goals using adapters supplied through config or constructor arguments.

    The action adapter decodes XYZ/XYZW pose, gripper, step budget, and move flag.
    It advances the pose reference and converts the eight pose/gripper values to native actions.
    """

    name = "agent_commanded_goal"

    def __init__(
        self,
        config: AgentCommandedGoalPolicyCfg,
        *,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: AgentActionAdapter | None = None,
    ):
        super().__init__(config, observation_adapter, action_adapter)
        self._goals = []
        self._grippers = []

    def get_action(self, env, observation):
        base = env.unwrapped
        assert self.task_description
        self.action_adapter.validate_environment(env)
        if not self._histories:
            self._histories = [deque(maxlen=1) for _ in range(base.num_envs)]
            self._steps = [0] * base.num_envs
            self._goals = [None] * base.num_envs
            self._grippers = [0.0] * base.num_envs
        assert len(self._goals) == base.num_envs
        # Calibration is only needed for decisions; ordinary tracking reads the same measured EEF pose.
        states = self.observation_adapter.extract_tracking_state(env, observation)
        calibrated_states = None
        actions = []
        for env_id, state in enumerate(states):
            measured = np.asarray(state["eef_pose_root_xyz_xyzw"])
            goal = self._goals[env_id]
            outcome = None
            if goal is not None:
                position_error, rotation_error = pose_error(goal["target"], measured)
                reference_error = pose_error(goal["target"], goal["reference"])
                converged = (
                    goal["move"]
                    and goal["executed"] >= self.config.min_convergence_steps
                    and position_error < self.config.position_convergence_m
                    and rotation_error < self.config.rotation_convergence_rad
                    and reference_error[0] < self.config.reference_position_tolerance_m
                    and reference_error[1] < self.config.reference_rotation_tolerance_rad
                )
                if converged or goal["executed"] >= goal["budget"]:
                    outcome = {
                        "outcome": "converged" if converged else ("incomplete" if goal["move"] else "completed"),
                        "executed_steps": goal["executed"],
                        "position_error_m": position_error,
                        "rotation_error_rad": rotation_error,
                        "target_xyz_xyzw": goal["target"].tolist(),
                        "gripper": self._grippers[env_id],
                    }
                    self._trace.write(
                        json.dumps({"event": "goal_result", "env_id": env_id, "step": self._steps[env_id], **outcome})
                        + "\n"
                    )
                    goal = None
            if goal is None:
                if calibrated_states is None:
                    calibrated_states = self.observation_adapter.extract_proprioception(env, observation)
                state = calibrated_states[env_id]
                state["last_gripper_command"] = self._grippers[env_id]
                state["episode_simulation_time_s"] = self._steps[env_id] * base.step_dt
                if outcome is not None:
                    state["previous_command_result"] = outcome
                frames = {
                    key: encode_image(observation["camera_obs"][key][env_id].cpu().numpy(), self.config.image_max_edge)
                    for key in self.observation_adapter.camera_keys
                }
                self._histories[env_id].append((self._steps[env_id], frames))
                command = self._infer(env_id, state)
                goal = {
                    "target": command[:7].astype(float),
                    "reference": measured.copy(),
                    "budget": int(command[8]),
                    "move": bool(command[9]),
                    "executed": 0,
                }
                self._goals[env_id] = goal
                self._grippers[env_id] = float(command[7])
            # Let the robot adapter compute this step's commanded pose toward the fixed goal.
            goal["reference"] = self.action_adapter.compute_next_reference_pose(goal["reference"], goal["target"])
            command = np.append(goal["reference"], self._grippers[env_id])
            action = self.action_adapter.command_to_action(env, command)
            if self.config.trace_actions:
                self._trace.write(
                    json.dumps({
                        "event": "execute",
                        "env_id": env_id,
                        "step": self._steps[env_id],
                        "proprioception": states[env_id],
                        "command": command.tolist(),
                        "native_action": action.cpu().tolist(),
                    })
                    + "\n"
                )
            actions.append(action)
            goal["executed"] += 1
            self._steps[env_id] += 1
        return torch.stack(actions)

    def reset(self, env_ids=None):
        if not self._histories:
            return
        ids = range(len(self._histories)) if env_ids is None else env_ids.reshape(-1).tolist()
        for env_id in ids:
            self._goals[env_id] = None
            self._grippers[env_id] = 0.0
        super().reset(env_ids)
