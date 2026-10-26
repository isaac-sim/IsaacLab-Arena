# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Control the fixed-base G1's left arm with a SpaceMouse.

The device integrates a pelvis-frame wrist target and emits a 23-D WBC-Pink action.
The right arm holds its home pose. Select it with ``--teleop_device spacemouse_g1_left_arm``.

Controls:
    Puck: translate and rotate the left wrist.
    Left button: toggle the left hand open/closed.
    Right button: reset the device and invoke the runner's registered ``R`` callback.
"""

from __future__ import annotations

import numpy as np
import time
import torch
from collections.abc import Callable
from scipy.spatial.transform import Rotation

from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device, register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_END_IDX,
)

# Wrist poses for the Bluefield open-arm reset posture: pelvis frame, quaternion XYZW.
LEFT_WRIST_HOME_POS = (0.1744, 0.2889, 0.1269)
LEFT_WRIST_HOME_QUAT_XYZW = (0.1208, -0.0308, 0.2454, 0.9614)
RIGHT_WRIST_HOLD_POS = (0.1744, -0.2889, 0.1269)
RIGHT_WRIST_HOLD_QUAT_XYZW = (-0.1208, -0.0308, -0.2454, 0.9614)

# Matches the environment's PELVIS_HEIGHT_ABOVE_GROUND (meters).
BASE_HEIGHT_CMD = 0.710

# Pelvis-frame XYZ target bounds (meters); these do not guarantee fingertip clearance.
LEFT_WRIST_POS_LIMITS = ((0.05, 0.60), (-0.30, 0.45), (0.05, 0.45))

# Full-deflection target speeds (m/s and rad/s), integrated using wall-clock time.
DEFAULT_POS_SENSITIVITY = 0.15
DEFAULT_ROT_SENSITIVITY = 0.75

# Cap elapsed time to prevent target jumps after loading or rendering stalls.
MAX_INTEGRATION_DT_S = 0.25

# Ignore deflections below this fraction of full scale to prevent idle drift.
PUCK_DEADZONE = 0.03

# Puck-to-pelvis XYZ signs for the front-facing Bluefield viewer.
PUCK_TO_PELVIS_SIGNS = (-1.0, -1.0, 1.0)


def _deadzoned(velocity: np.ndarray, full_scale: float) -> np.ndarray:
    """Zero out components of ``velocity`` below ``PUCK_DEADZONE`` of ``full_scale``."""
    out = np.asarray(velocity, dtype=float).copy()
    out[np.abs(out) < PUCK_DEADZONE * full_scale] = 0.0
    return out


class G1LeftArmSpaceMouse(Se3SpaceMouse):
    """Integrate SpaceMouse velocities into a left-wrist target using the base HID listener.

    Sensitivities represent speeds; ``advance()`` integrates the scaled input buffers.
    """

    def __init__(self, cfg: Se3SpaceMouseCfg):
        """Initialize the device at the left wrist's home pose.

        Args:
            cfg: SpaceMouse settings with sensitivities expressed as speeds.
        """
        # Set before super().__init__(), which starts the listener thread that can call reset().
        self._left_pos = np.array(LEFT_WRIST_HOME_POS)
        self._left_rot = Rotation.from_quat(LEFT_WRIST_HOME_QUAT_XYZW)
        self._last_advance_time: float | None = None
        super().__init__(cfg)

    def __str__(self) -> str:
        """Describe the device and its button mapping."""
        return (
            f"{super().__str__()}\n"
            "\t----------------------------------------------\n"
            "\tOnly the left arm moves; the right arm holds its spawn pose.\n"
            "\tLeft button: toggle the left hand open/closed"
        )

    def reset(self):
        """Recentre the left-wrist target on the home pose and open the hand."""
        super().reset()
        self._left_pos = np.array(LEFT_WRIST_HOME_POS)
        self._left_rot = Rotation.from_quat(LEFT_WRIST_HOME_QUAT_XYZW)
        # Exclude reset time from the next integration step.
        self._last_advance_time = None
        # Re-enable rotation packets after the base class's two-button toggle.
        self._read_rotation = False

    def _integration_dt(self) -> float:
        """Seconds elapsed since the previous ``advance()``, clamped, or 0.0 on the first call."""
        now = time.perf_counter()
        previous = self._last_advance_time
        self._last_advance_time = now
        if previous is None:
            return 0.0
        return min(now - previous, MAX_INTEGRATION_DT_S)

    def advance(self) -> torch.Tensor:
        """Integrate the puck's velocity into the left-wrist target and return a 23-D action."""
        dt = self._integration_dt()
        delta_pos = PUCK_TO_PELVIS_SIGNS * _deadzoned(self._delta_pos, self.pos_sensitivity) * dt
        delta_rot = PUCK_TO_PELVIS_SIGNS * _deadzoned(self._delta_rot, self.rot_sensitivity) * dt
        self._left_pos = np.clip(self._left_pos + delta_pos, *np.transpose(LEFT_WRIST_POS_LIMITS))
        if np.any(delta_rot):
            # Pre-multiply to rotate about pelvis-frame axes.
            self._left_rot = Rotation.from_euler("XYZ", delta_rot) * self._left_rot

        action = np.zeros(TORSO_ORIENTATION_RPY_CMD_END_IDX)
        action[LEFT_HAND_STATE_IDX] = 1.0 if self._close_gripper else 0.0
        action[LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = self._left_pos
        action[LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = self._left_rot.as_quat()
        action[RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = RIGHT_WRIST_HOLD_POS
        action[RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = RIGHT_WRIST_HOLD_QUAT_XYZW
        action[BASE_HEIGHT_CMD_START_IDX] = BASE_HEIGHT_CMD
        # Right hand, navigation and torso commands remain zero.
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)


@configclass
class G1LeftArmSpaceMouseCfg(Se3SpaceMouseCfg):
    """Configure left-arm SpaceMouse control for the fixed-base G1."""

    pos_sensitivity: float = DEFAULT_POS_SENSITIVITY
    """Wrist-target speed at full puck deflection (m/s)."""
    rot_sensitivity: float = DEFAULT_ROT_SENSITIVITY
    """Wrist-target angular speed at full puck deflection (rad/s)."""
    gripper_term: bool = False
    """Disable the base gripper term; ``advance()`` fills the action's hand-state field."""
    class_type: type[Se3SpaceMouse] = G1LeftArmSpaceMouse


@register_device
class G1LeftArmSpaceMouseDevice(TeleopDeviceBase):
    """Register ``spacemouse_g1_left_arm`` and its configurable target speeds."""

    name = "spacemouse_g1_left_arm"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = DEFAULT_POS_SENSITIVITY,
        rot_sensitivity: float = DEFAULT_ROT_SENSITIVITY,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> G1LeftArmSpaceMouseCfg:
        return G1LeftArmSpaceMouseCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_retargeter
class G1FixedBaseSpaceMouseRetargeter(RetargetterBase):
    """Pair the device with the fixed-base G1; its WBC-Pink actions need no retargeting pipeline."""

    device = "spacemouse_g1_left_arm"
    embodiment = "g1_fixed_base_upper_body"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
