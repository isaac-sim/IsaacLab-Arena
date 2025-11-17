# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Keyboard to 23D G1 WBC action adapter.

This adapter allows keyboard control of the full 23-dimensional action space
for the Unitree G1 humanoid robot with Whole Body Controller (WBC).
"""

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from enum import Enum

import carb
import omni


class ControlMode(Enum):
    """Control modes for the keyboard adapter."""
    RIGHT_HAND = "right_hand"
    LEFT_HAND = "left_hand"
    BOTH_HANDS = "both_hands"
    BASE_NAV = "base_nav"
    TORSO = "torso"
    HEIGHT = "height"


@dataclass
class KeyboardTo23DConfig:
    """Configuration for keyboard to 23D adapter."""
    
    # Sensitivity settings
    pos_sensitivity: float = 0.01  # meters per key press
    rot_sensitivity: float = 0.05  # radians per key press
    vel_sensitivity: float = 0.05  # m/s per key press
    height_sensitivity: float = 0.01  # meters per key press
    
    # Default values
    default_base_height: float = 0.75  # meters
    default_left_hand_pos: list[float] = None  # [x, y, z]
    default_right_hand_pos: list[float] = None  # [x, y, z]
    
    # Limits
    max_velocity: float = 0.4  # m/s
    min_base_height: float = 0.65  # meters
    max_base_height: float = 0.85  # meters
    max_torso_angle: float = 0.3  # radians
    
    # Hand position limits (workspace bounds for G1)
    hand_pos_x_min: float = -0.1  # meters (allow slight backward reach for balance)
    hand_pos_x_max: float = 0.5   # meters (forward reach)
    hand_pos_y_min: float = -0.5  # meters (right limit)
    hand_pos_y_max: float = 0.5   # meters (left limit)
    hand_pos_z_min: float = 0.1   # meters (avoid ground collision)
    hand_pos_z_max: float = 0.5   # meters (max height)
    
    def __post_init__(self):
        if self.default_left_hand_pos is None:
            # G1 default: hands forward at waist level, within workspace
            self.default_left_hand_pos = [0.15, 0.15, 0.2]
        if self.default_right_hand_pos is None:
            self.default_right_hand_pos = [0.15, -0.15, 0.2]


class KeyboardTo23DAdapter:
    """Adapter that converts keyboard input to 23D G1 WBC actions.
    
    This adapter uses mode switching to control different aspects of the robot:
    - Both hands (synchronized control)
    - Right hand (position, orientation, gripper)
    - Left hand (position, orientation, gripper)
    - Base navigation (linear/angular velocity)
    - Torso orientation (roll, pitch, yaw)
    - Base height
    
    Key bindings:
        Mode switching:
            0: Both hands mode (synchronized)
            1: Right hand mode
            2: Left hand mode
            3: Base navigation mode
            4: Torso orientation mode
            5: Base height mode
        
        Both hands mode (synchronized control):
            Q/E: Both hands up/down
            A/D: Hands apart/together (spread/close)
            W/S: Both hands forward/backward
            Z/X: Symmetric wrist roll (hands mirror each other)
            K: Toggle both grippers
        
        Hand mode (when in right/left hand mode):
            W/S: Move forward/backward (X)
            A/D: Move left/right (Y)
            Q/E: Move up/down (Z)
            Z/X: Roll rotation
            T/G: Pitch rotation
            C/V: Yaw rotation
            K: Toggle gripper
        
        Base navigation mode:
            W/S: Forward/backward velocity
            A/D: Left/right velocity
            Q/E: Rotate left/right
            X: Emergency stop (zero all velocities)
        
        Torso mode:
            Z/X: Roll left/right
            T/G: Pitch forward/backward
            C/V: Yaw left/right
        
        Height mode:
            W/S: Raise/lower base
        
        Special keys:
            R: Reset all to default
            L: Lock/unlock current dimension
            Space: Pause/resume updates
            Esc: (handled by environment)
    """
    
    def __init__(self, cfg: KeyboardTo23DConfig, sim_device: str = "cpu"):
        """Initialize the adapter.
        
        Args:
            cfg: Configuration object
            sim_device: Device for torch tensors ('cpu' or 'cuda')
        """
        self.cfg = cfg
        self._sim_device = sim_device
        
        # Current control mode
        self.mode = ControlMode.BOTH_HANDS
        
        # Pause state
        self.paused = False
        
        # 23D state storage
        self._state = {
            'left_hand': 0.0,  # -1.0 (closed) to 1.0 (open)
            'right_hand': 0.0,
            'left_wrist_pos': np.array(cfg.default_left_hand_pos, dtype=np.float32),
            'left_wrist_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # w, x, y, z
            'right_wrist_pos': np.array(cfg.default_right_hand_pos, dtype=np.float32),
            'right_wrist_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'navigate_cmd': np.array([0.0, 0.0, 0.0], dtype=np.float32),  # vx, vy, omega_z
            'base_height': cfg.default_base_height,
            'torso_rpy': np.array([0.0, 0.0, 0.0], dtype=np.float32),  # roll, pitch, yaw
        }
        
        # Locked dimensions (not updated)
        self.locked_dims = set()
        
        # Additional callbacks
        self._additional_callbacks = dict()
        
        # Setup keyboard interface
        self._setup_keyboard()
        
        # Print initial instructions
        self._print_instructions()
    
    def _setup_keyboard(self):
        """Setup keyboard event listener."""
        import weakref
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
    
    def __del__(self):
        """Cleanup keyboard interface."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
    
    def __str__(self) -> str:
        """Return description string."""
        msg = f"Keyboard to 23D Adapter for G1 WBC\n"
        msg += f"Current mode: {self.mode.value}\n"
        msg += f"----------------------------------------------\n"
        msg += f"Mode switching: 0=BothHands 1=RightHand 2=LeftHand 3=BaseNav 4=Torso 5=Height\n"
        msg += f"Special: R=Reset L=Lock Space=Pause\n"
        return msg
    
    def _print_instructions(self):
        """Print usage instructions."""
        print("=" * 60)
        print("Keyboard to 23D Adapter - Controls")
        print("=" * 60)
        print("MODE SWITCHING:")
        print("  [0] Both Hands  [1] Right Hand  [2] Left Hand")
        print("  [3] Base Nav    [4] Torso       [5] Height")
        print("")
        print("BOTH HANDS MODE (Synchronized):")
        print("  Q/E: Both Up/Down   A/D: Apart/Together")
        print("  W/S: Both Forward/Back   Z/X: Symmetric Roll")
        print("  K: Toggle Both Grippers")
        print("")
        print("HAND MODE (Right/Left):")
        print("  W/S: Forward/Back   A/D: Left/Right   Q/E: Up/Down")
        print("  Z/X: Roll   T/G: Pitch   C/V: Yaw   K: Gripper")
        print("")
        print("BASE NAVIGATION MODE:")
        print("  W/S: Forward/Back   A/D: Strafe   Q/E: Rotate   X: STOP")
        print("")
        print("TORSO MODE:")
        print("  Z/X: Roll   T/G: Pitch   C/V: Yaw")
        print("")
        print("HEIGHT MODE:")
        print("  W/S: Raise/Lower")
        print("")
        print("SPECIAL KEYS:")
        print("  [R] Reset all   [L] Lock mode   [Space] Pause")
        print("")
        print("WORKSPACE LIMITS:")
        print(f"  Hand X: [{self.cfg.hand_pos_x_min:.2f}, {self.cfg.hand_pos_x_max:.2f}] m")
        print(f"  Hand Y: [{self.cfg.hand_pos_y_min:.2f}, {self.cfg.hand_pos_y_max:.2f}] m")
        print(f"  Hand Z: [{self.cfg.hand_pos_z_min:.2f}, {self.cfg.hand_pos_z_max:.2f}] m")
        print(f"  Base velocity: ±{self.cfg.max_velocity:.2f} m/s")
        print(f"  Base height: [{self.cfg.min_base_height:.2f}, {self.cfg.max_base_height:.2f}] m")
        print("=" * 60)
    
    def reset(self):
        """Reset all states to default."""
        self._state = {
            'left_hand': 0.0,
            'right_hand': 0.0,
            'left_wrist_pos': np.array(self.cfg.default_left_hand_pos, dtype=np.float32),
            'left_wrist_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'right_wrist_pos': np.array(self.cfg.default_right_hand_pos, dtype=np.float32),
            'right_wrist_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'navigate_cmd': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'base_height': self.cfg.default_base_height,
            'torso_rpy': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
        self.locked_dims.clear()
        self.paused = False
        print(f"[Adapter] Reset to default state")
        self._print_status()
    
    def add_callback(self, key: str, func: Callable):
        """Add additional callback for specific key.
        
        Args:
            key: Key name (e.g., 'ESC', 'ENTER')
            func: Callback function (no arguments)
        """
        self._additional_callbacks[key] = func
    
    def advance(self) -> torch.Tensor:
        """Get current 23D action as tensor.
        
        Returns:
            torch.Tensor: 23D action tensor
        """
        if self.paused:
            # Return current state when paused (maintain pose, zero velocities)
            action = np.zeros(23, dtype=np.float32)
            action[0] = self._state['left_hand']
            action[1] = self._state['right_hand']
            action[2:5] = self._state['left_wrist_pos']
            action[5:9] = self._state['left_wrist_quat']
            action[9:12] = self._state['right_wrist_pos']
            action[12:16] = self._state['right_wrist_quat']
            action[16:19] = np.array([0.0, 0.0, 0.0])  # Zero navigation velocities
            action[19] = self._state['base_height']
            action[20:23] = self._state['torso_rpy']
            return torch.tensor(action, dtype=torch.float32, device=self._sim_device)
        
        action = np.zeros(23, dtype=np.float32)
        
        # [0-1] Hand states
        action[0] = self._state['left_hand']
        action[1] = self._state['right_hand']
        
        # [2-8] Left wrist (pos + quat)
        action[2:5] = self._state['left_wrist_pos']
        action[5:9] = self._state['left_wrist_quat']
        
        # [9-15] Right wrist (pos + quat)
        action[9:12] = self._state['right_wrist_pos']
        action[12:16] = self._state['right_wrist_quat']
        
        # [16-18] Navigation command
        action[16:19] = self._state['navigate_cmd']
        
        # [19] Base height
        action[19] = self._state['base_height']
        
        # [20-22] Torso orientation
        action[20:23] = self._state['torso_rpy']
        
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            
            # Mode switching (use KEY_0, KEY_1, KEY_2, etc. for number keys)
            if key_name == "KEY_0":
                self.mode = ControlMode.BOTH_HANDS
                print(f"[Mode] Switched to BOTH HANDS (Synchronized)")
                self._print_status()
            elif key_name == "KEY_1":
                self.mode = ControlMode.RIGHT_HAND
                print(f"[Mode] Switched to RIGHT HAND")
                self._print_status()
            elif key_name == "KEY_2":
                self.mode = ControlMode.LEFT_HAND
                print(f"[Mode] Switched to LEFT HAND")
                self._print_status()
            elif key_name == "KEY_3":
                self.mode = ControlMode.BASE_NAV
                print(f"[Mode] Switched to BASE NAVIGATION")
                self._print_status()
            elif key_name == "KEY_4":
                self.mode = ControlMode.TORSO
                print(f"[Mode] Switched to TORSO ORIENTATION")
                self._print_status()
            elif key_name == "KEY_5":
                self.mode = ControlMode.HEIGHT
                print(f"[Mode] Switched to HEIGHT CONTROL")
                self._print_status()
            
            # Special keys
            elif key_name == "R":
                self.reset()
            elif key_name == "L":
                mode_key = self.mode.value
                if mode_key in self.locked_dims:
                    self.locked_dims.remove(mode_key)
                    print(f"[Lock] Unlocked {self.mode.value}")
                else:
                    self.locked_dims.add(mode_key)
                    print(f"[Lock] Locked {self.mode.value}")
            elif key_name == "SPACE":
                self.paused = not self.paused
                print(f"[Pause] {'PAUSED' if self.paused else 'RESUMED'}")
            
            # Mode-specific controls (only if not locked)
            elif self.mode.value not in self.locked_dims and not self.paused:
                self._process_mode_input(key_name)
            
            # Additional callbacks
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()
        
        return True
    
    def _process_mode_input(self, key_name: str):
        """Process input based on current mode."""
        if self.mode == ControlMode.BOTH_HANDS:
            self._process_both_hands_input(key_name)
        elif self.mode == ControlMode.RIGHT_HAND:
            self._process_hand_input(key_name, hand='right')
        elif self.mode == ControlMode.LEFT_HAND:
            self._process_hand_input(key_name, hand='left')
        elif self.mode == ControlMode.BASE_NAV:
            self._process_base_nav_input(key_name)
        elif self.mode == ControlMode.TORSO:
            self._process_torso_input(key_name)
        elif self.mode == ControlMode.HEIGHT:
            self._process_height_input(key_name)
    
    def _process_both_hands_input(self, key_name: str):
        """Process both hands synchronized control input.
        
        Key mappings:
            Q/E: Both hands up/down (Z axis)
            A/D: Hands apart/together (Y axis, symmetric)
            W/S: Both hands forward/backward (X axis)
            Z/X: Symmetric wrist roll (left and right hands roll in opposite directions for mirrored motion)
            K: Toggle both grippers
        """
        # Q/E: Both hands up/down
        if key_name == "Q":
            # Both hands move up
            self._state['left_wrist_pos'][2] = min(
                self._state['left_wrist_pos'][2] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_max
            )
            self._state['right_wrist_pos'][2] = min(
                self._state['right_wrist_pos'][2] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_max
            )
        elif key_name == "E":
            # Both hands move down
            self._state['left_wrist_pos'][2] = max(
                self._state['left_wrist_pos'][2] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_min
            )
            self._state['right_wrist_pos'][2] = max(
                self._state['right_wrist_pos'][2] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_min
            )
        
        # A/D: Hands apart/together (symmetric Y movement)
        elif key_name == "A":
            # Hands move apart (left hand goes left +Y, right hand goes right -Y)
            self._state['left_wrist_pos'][1] = min(
                self._state['left_wrist_pos'][1] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_max
            )
            self._state['right_wrist_pos'][1] = max(
                self._state['right_wrist_pos'][1] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_min
            )
        elif key_name == "D":
            # Hands move together (left hand goes right -Y, right hand goes left +Y)
            self._state['left_wrist_pos'][1] = max(
                self._state['left_wrist_pos'][1] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_min
            )
            self._state['right_wrist_pos'][1] = min(
                self._state['right_wrist_pos'][1] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_max
            )
        
        # W/S: Both hands forward/backward (X axis)
        
        elif key_name == "W":
            # Both hands move forward
            self._state['left_wrist_pos'][0] = min(
                self._state['left_wrist_pos'][0] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_max
            )
            self._state['right_wrist_pos'][0] = min(
                self._state['right_wrist_pos'][0] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_max
            )
        elif key_name == "S":
            # Both hands move backward
            self._state['left_wrist_pos'][0] = max(
                self._state['left_wrist_pos'][0] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_min
            )
            self._state['right_wrist_pos'][0] = max(
                self._state['right_wrist_pos'][0] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_min
            )
        
        # Z/X: Wrist rotation (roll, since pitch doesn't work with WBC)
        elif key_name == "Z":
            # Apply symmetric roll: left hand positive, right hand negative (for mirrored motion)
            delta_rpy_left = np.array([self.cfg.rot_sensitivity, 0.0, 0.0])
            delta_rpy_right = np.array([-self.cfg.rot_sensitivity, 0.0, 0.0])  # Opposite for symmetry
            
            # Apply to left hand
            delta_quat_left = Rotation.from_euler('xyz', delta_rpy_left).as_quat()
            delta_quat_left_wxyz = np.array([delta_quat_left[3], delta_quat_left[0], delta_quat_left[1], delta_quat_left[2]])
            old_quat_left = self._state['left_wrist_quat'].copy()
            new_quat_left = self._quaternion_multiply(self._state['left_wrist_quat'], delta_quat_left_wxyz)
            norm_left = np.linalg.norm(new_quat_left)
            if norm_left > 1e-6:
                self._state['left_wrist_quat'] = new_quat_left / norm_left
            
            # Apply to right hand (opposite roll)
            delta_quat_right = Rotation.from_euler('xyz', delta_rpy_right).as_quat()
            delta_quat_right_wxyz = np.array([delta_quat_right[3], delta_quat_right[0], delta_quat_right[1], delta_quat_right[2]])
            old_quat_right = self._state['right_wrist_quat'].copy()
            new_quat_right = self._quaternion_multiply(self._state['right_wrist_quat'], delta_quat_right_wxyz)
            norm_right = np.linalg.norm(new_quat_right)
            if norm_right > 1e-6:
                self._state['right_wrist_quat'] = new_quat_right / norm_right
            
            print(f"[Both Hands] Z pressed - Symmetric Roll (L:+{self.cfg.rot_sensitivity:.3f} R:-{self.cfg.rot_sensitivity:.3f} rad)")
            print(f"  Left quat:  {old_quat_left} → {self._state['left_wrist_quat']}")
            print(f"  Right quat: {old_quat_right} → {self._state['right_wrist_quat']}")
                
        elif key_name == "X":
            # Apply symmetric roll (opposite direction): left hand negative, right hand positive
            delta_rpy_left = np.array([-self.cfg.rot_sensitivity, 0.0, 0.0])
            delta_rpy_right = np.array([self.cfg.rot_sensitivity, 0.0, 0.0])  # Opposite for symmetry
            
            # Apply to left hand
            delta_quat_left = Rotation.from_euler('xyz', delta_rpy_left).as_quat()
            delta_quat_left_wxyz = np.array([delta_quat_left[3], delta_quat_left[0], delta_quat_left[1], delta_quat_left[2]])
            old_quat_left = self._state['left_wrist_quat'].copy()
            new_quat_left = self._quaternion_multiply(self._state['left_wrist_quat'], delta_quat_left_wxyz)
            norm_left = np.linalg.norm(new_quat_left)
            if norm_left > 1e-6:
                self._state['left_wrist_quat'] = new_quat_left / norm_left
            
            # Apply to right hand (opposite roll)
            delta_quat_right = Rotation.from_euler('xyz', delta_rpy_right).as_quat()
            delta_quat_right_wxyz = np.array([delta_quat_right[3], delta_quat_right[0], delta_quat_right[1], delta_quat_right[2]])
            old_quat_right = self._state['right_wrist_quat'].copy()
            new_quat_right = self._quaternion_multiply(self._state['right_wrist_quat'], delta_quat_right_wxyz)
            norm_right = np.linalg.norm(new_quat_right)
            if norm_right > 1e-6:
                self._state['right_wrist_quat'] = new_quat_right / norm_right
            
            print(f"[Both Hands] X pressed - Symmetric Roll (L:-{self.cfg.rot_sensitivity:.3f} R:+{self.cfg.rot_sensitivity:.3f} rad)")
            print(f"  Left quat:  {old_quat_left} → {self._state['left_wrist_quat']}")
            print(f"  Right quat: {old_quat_right} → {self._state['right_wrist_quat']}")
        
        # K: Toggle both grippers
        elif key_name == "K":
            self._state['left_hand'] = -self._state['left_hand']
            self._state['right_hand'] = -self._state['right_hand']
            status = "CLOSED" if self._state['left_hand'] < 0 else "OPEN"
            print(f"[Both Hands] Grippers: {status}")
    
    def _process_hand_input(self, key_name: str, hand: str):
        """Process hand control input."""
        pos_key = f'{hand}_wrist_pos'
        quat_key = f'{hand}_wrist_quat'
        gripper_key = f'{hand}_hand'
        
        # Position control (with limits)
        if key_name == "W":
            self._state[pos_key][0] = min(
                self._state[pos_key][0] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_max
            )
        elif key_name == "S":
            self._state[pos_key][0] = max(
                self._state[pos_key][0] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_x_min
            )
        elif key_name == "A":
            self._state[pos_key][1] = min(
                self._state[pos_key][1] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_max
            )
        elif key_name == "D":
            self._state[pos_key][1] = max(
                self._state[pos_key][1] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_y_min
            )
        elif key_name == "Q":
            self._state[pos_key][2] = min(
                self._state[pos_key][2] + self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_max
            )
        elif key_name == "E":
            self._state[pos_key][2] = max(
                self._state[pos_key][2] - self.cfg.pos_sensitivity,
                self.cfg.hand_pos_z_min
            )
        
        # Rotation control (apply incremental rotation)
        elif key_name in ["Z", "X", "T", "G", "C", "V"]:
            delta_rpy = np.zeros(3)
            if key_name == "Z":
                delta_rpy[0] = self.cfg.rot_sensitivity  # Roll +
            elif key_name == "X":
                delta_rpy[0] = -self.cfg.rot_sensitivity  # Roll -
            elif key_name == "T":
                delta_rpy[1] = self.cfg.rot_sensitivity  # Pitch +
            elif key_name == "G":
                delta_rpy[1] = -self.cfg.rot_sensitivity  # Pitch -
            elif key_name == "C":
                delta_rpy[2] = self.cfg.rot_sensitivity  # Yaw +
            elif key_name == "V":
                delta_rpy[2] = -self.cfg.rot_sensitivity  # Yaw -
            
            # Apply rotation: new_quat = current_quat * delta_quat
            delta_quat = Rotation.from_euler('xyz', delta_rpy).as_quat()  # x,y,z,w format
            delta_quat_wxyz = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])
            current_quat_wxyz = self._state[quat_key]
            new_quat = self._quaternion_multiply(current_quat_wxyz, delta_quat_wxyz)
            
            # Normalize with safety check
            norm = np.linalg.norm(new_quat)
            if norm > 1e-6:  # Avoid division by zero
                self._state[quat_key] = new_quat / norm
            else:
                # Reset to identity quaternion if corrupted
                self._state[quat_key] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                print(f"[Warning] {hand.capitalize()} hand quaternion reset to identity")
        
        # Gripper control
        elif key_name == "K":
            self._state[gripper_key] = -self._state[gripper_key]  # Toggle between -1 and 1
            status = "CLOSED" if self._state[gripper_key] < 0 else "OPEN"
            print(f"[{hand.capitalize()} Hand] Gripper: {status}")
    
    def _process_base_nav_input(self, key_name: str):
        """Process base navigation input."""
        if key_name == "W":
            self._state['navigate_cmd'][0] = min(
                self._state['navigate_cmd'][0] + self.cfg.vel_sensitivity,
                self.cfg.max_velocity
            )
        elif key_name == "S":
            self._state['navigate_cmd'][0] = max(
                self._state['navigate_cmd'][0] - self.cfg.vel_sensitivity,
                -self.cfg.max_velocity
            )
        elif key_name == "A":
            self._state['navigate_cmd'][1] = min(
                self._state['navigate_cmd'][1] + self.cfg.vel_sensitivity,
                self.cfg.max_velocity
            )
        elif key_name == "D":
            self._state['navigate_cmd'][1] = max(
                self._state['navigate_cmd'][1] - self.cfg.vel_sensitivity,
                -self.cfg.max_velocity
            )
        elif key_name == "Q":
            self._state['navigate_cmd'][2] = min(
                self._state['navigate_cmd'][2] + self.cfg.vel_sensitivity,
                1.0
            )
        elif key_name == "E":
            self._state['navigate_cmd'][2] = max(
                self._state['navigate_cmd'][2] - self.cfg.vel_sensitivity,
                -1.0
            )
        elif key_name == "X":
            # Emergency stop: zero all navigation velocities
            self._state['navigate_cmd'] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            print("[Navigation] STOP - All velocities zeroed")
    
    def _process_torso_input(self, key_name: str):
        """Process torso orientation input."""
        if key_name == "Z":
            self._state['torso_rpy'][0] = min(
                self._state['torso_rpy'][0] + self.cfg.rot_sensitivity,
                self.cfg.max_torso_angle
            )
        elif key_name == "X":
            self._state['torso_rpy'][0] = max(
                self._state['torso_rpy'][0] - self.cfg.rot_sensitivity,
                -self.cfg.max_torso_angle
            )
        elif key_name == "T":
            self._state['torso_rpy'][1] = min(
                self._state['torso_rpy'][1] + self.cfg.rot_sensitivity,
                self.cfg.max_torso_angle
            )
        elif key_name == "G":
            self._state['torso_rpy'][1] = max(
                self._state['torso_rpy'][1] - self.cfg.rot_sensitivity,
                -self.cfg.max_torso_angle
            )
        elif key_name == "C":
            self._state['torso_rpy'][2] = min(
                self._state['torso_rpy'][2] + self.cfg.rot_sensitivity,
                self.cfg.max_torso_angle
            )
        elif key_name == "V":
            self._state['torso_rpy'][2] = max(
                self._state['torso_rpy'][2] - self.cfg.rot_sensitivity,
                -self.cfg.max_torso_angle
            )
    
    def _process_height_input(self, key_name: str):
        """Process height control input."""
        if key_name == "W":
            self._state['base_height'] = min(
                self._state['base_height'] + self.cfg.height_sensitivity,
                self.cfg.max_base_height
            )
        elif key_name == "S":
            self._state['base_height'] = max(
                self._state['base_height'] - self.cfg.height_sensitivity,
                self.cfg.min_base_height
            )
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions (w, x, y, z format).
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
        
        Returns:
            Result quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _print_status(self):
        """Print current state status."""
        print("─" * 60)
        print(f"Current Mode: {self.mode.value.upper()} {'🔒' if self.mode.value in self.locked_dims else ''}")
        print(f"Left Hand:  Pos({self._state['left_wrist_pos'][0]:.2f}, {self._state['left_wrist_pos'][1]:.2f}, {self._state['left_wrist_pos'][2]:.2f})  {'CLOSED' if self._state['left_hand'] < 0 else 'OPEN'}")
        print(f"Right Hand: Pos({self._state['right_wrist_pos'][0]:.2f}, {self._state['right_wrist_pos'][1]:.2f}, {self._state['right_wrist_pos'][2]:.2f})  {'CLOSED' if self._state['right_hand'] < 0 else 'OPEN'}")
        print(f"Navigation: vx={self._state['navigate_cmd'][0]:.2f} vy={self._state['navigate_cmd'][1]:.2f} ω={self._state['navigate_cmd'][2]:.2f}")
        print(f"Base:       Height={self._state['base_height']:.2f}m")
        print(f"Torso:      R={self._state['torso_rpy'][0]:.2f} P={self._state['torso_rpy'][1]:.2f} Y={self._state['torso_rpy'][2]:.2f}")
        print("─" * 60)

