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

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


# Global state tracker for multi-stage tasks
# Key: env_id, Value: dict of stage completion flags
_MULTI_STAGE_STATE = {}


# NOTE(alexmillane, 2025.09.15): The velocity threshold is set high because some stationary
# seem to generate a "small" velocity.
def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    sensor: ContactSensor = env.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(sensor.data.force_matrix_w.clone(), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = object.data.root_lin_vel_w
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)
    return condition_met


def objects_in_proximity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns:
        Boolean tensor indicating when objects are within a certain proximity of each other.
    """
    # Get object entities from the scene
    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins

    # Get positions relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins
    target_object_pos = target_object.data.root_pos_w - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def multi_stage_pick_place_and_navigate(
    env: ManagerBasedRLEnv,
    # Stage 1: Box on cart
    box_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    box_to_cart_max_x: float,
    box_to_cart_max_y: float,
    box_to_cart_max_z: float,
    # Stage 2: Cart to destination
    destination_cfg: SceneEntityCfg,
    cart_to_dest_max_x: float,
    cart_to_dest_max_y: float,
    cart_to_dest_max_z: float,
) -> torch.Tensor:
    """
    Two-stage success check (position-based only):
    Stage 1: Box must be on cart
    Stage 2: Cart must be at destination
    
    Success only when BOTH stages are complete!
    
    Args:
        env: Environment
        box_cfg: Config for the box object
        cart_cfg: Config for the cart object
        destination_cfg: Config for the destination (e.g., surgical bed)
        box_to_cart_max_x/y/z: Max separation for box-cart (Stage 1)
        cart_to_dest_max_x/y/z: Max separation for cart-destination (Stage 2)
    
    Returns:
        Boolean tensor: True only when both stages are complete
    """
    global _MULTI_STAGE_STATE
    
    # Get objects
    box: RigidObject = env.scene[box_cfg.name]
    cart: RigidObject = env.scene[cart_cfg.name]
    destination: RigidObject = env.scene[destination_cfg.name]
    
    # Get positions relative to environment origin
    box_pos = box.data.root_pos_w - env.scene.env_origins
    cart_pos = cart.data.root_pos_w - env.scene.env_origins
    dest_pos = destination.data.root_pos_w - env.scene.env_origins
    
    # Number of environments
    num_envs = box_pos.shape[0]
    
    # Initialize state if needed
    if "stage1_complete" not in _MULTI_STAGE_STATE:
        _MULTI_STAGE_STATE["stage1_complete"] = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    if "stage2_complete" not in _MULTI_STAGE_STATE:
        _MULTI_STAGE_STATE["stage2_complete"] = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    
    # --- Stage 1: Box on Cart (position only) ---
    box_cart_x_sep = torch.abs(box_pos[:, 0] - cart_pos[:, 0])
    box_cart_y_sep = torch.abs(box_pos[:, 1] - cart_pos[:, 1])
    box_cart_z_sep = torch.abs(box_pos[:, 2] - cart_pos[:, 2])
    
    stage1_current = box_cart_x_sep < box_to_cart_max_x
    stage1_current = torch.logical_and(stage1_current, box_cart_y_sep < box_to_cart_max_y)
    stage1_current = torch.logical_and(stage1_current, box_cart_z_sep < box_to_cart_max_z)
    
    # Update stage1 state (once complete, stays complete)
    _MULTI_STAGE_STATE["stage1_complete"] = torch.logical_or(
        _MULTI_STAGE_STATE["stage1_complete"], 
        stage1_current
    )
    
    # --- Stage 2: Cart to Destination (position only) ---
    cart_dest_x_sep = torch.abs(cart_pos[:, 0] - dest_pos[:, 0])
    cart_dest_y_sep = torch.abs(cart_pos[:, 1] - dest_pos[:, 1])
    cart_dest_z_sep = torch.abs(cart_pos[:, 2] - dest_pos[:, 2])
    
    stage2_current = cart_dest_x_sep < cart_to_dest_max_x
    stage2_current = torch.logical_and(stage2_current, cart_dest_y_sep < cart_to_dest_max_y)
    stage2_current = torch.logical_and(stage2_current, cart_dest_z_sep < cart_to_dest_max_z)
    
    # Update stage2 state (once complete, stays complete)
    _MULTI_STAGE_STATE["stage2_complete"] = torch.logical_or(
        _MULTI_STAGE_STATE["stage2_complete"],
        stage2_current
    )
    
    # --- Success: Both stages complete ---
    both_stages_complete = torch.logical_and(
        _MULTI_STAGE_STATE["stage1_complete"],
        _MULTI_STAGE_STATE["stage2_complete"]
    )
    
    return both_stages_complete


def reset_multi_stage_state():
    """Reset the multi-stage state tracker. Call this when environment resets."""
    global _MULTI_STAGE_STATE
    _MULTI_STAGE_STATE.clear()


def multi_stage_with_fixed_destination(
    env: ManagerBasedRLEnv,
    # Stage 1: Box on cart
    box_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    box_to_cart_max_x: float,
    box_to_cart_max_y: float,
    box_to_cart_max_z: float,
    # Stage 2: Cart to fixed position (surgical bed location)
    target_position_x: float,
    target_position_y: float,
    target_position_z: float,
    cart_to_target_max_x: float,
    cart_to_target_max_y: float,
    cart_to_target_max_z: float,
) -> torch.Tensor:
    """
    Two-stage success with fixed target position:
    Stage 1: Box must be on cart
    Stage 2: Cart must be at fixed target position (e.g., surgical bed coordinates)
    
    Success only when BOTH stages are complete!
    
    Args:
        env: Environment
        box_cfg: Config for the box object
        cart_cfg: Config for the cart object
        box_to_cart_max_x/y/z: Max separation for box-cart (Stage 1)
        target_position_x/y/z: Fixed target coordinates (e.g., surgical bed location)
        cart_to_target_max_x/y/z: Max separation for cart-target (Stage 2)
    
    Returns:
        Boolean tensor: True only when both stages are complete
    """
    global _MULTI_STAGE_STATE
    
    # Get objects
    box: RigidObject = env.scene[box_cfg.name]
    cart: RigidObject = env.scene[cart_cfg.name]
    
    # Get positions relative to environment origin
    box_pos = box.data.root_pos_w - env.scene.env_origins
    cart_pos = cart.data.root_pos_w - env.scene.env_origins
    
    # Target position (fixed coordinates)
    target_pos = torch.tensor([target_position_x, target_position_y, target_position_z], 
                              device=env.device).unsqueeze(0)  # Shape: (1, 3)
    
    # Number of environments
    num_envs = box_pos.shape[0]
    
    # Initialize state if needed
    if "stage1_complete" not in _MULTI_STAGE_STATE:
        _MULTI_STAGE_STATE["stage1_complete"] = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    if "stage2_complete" not in _MULTI_STAGE_STATE:
        _MULTI_STAGE_STATE["stage2_complete"] = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    
    # --- Stage 1: Box on Cart ---
    box_cart_x_sep = torch.abs(box_pos[:, 0] - cart_pos[:, 0])
    box_cart_y_sep = torch.abs(box_pos[:, 1] - cart_pos[:, 1])
    box_cart_z_sep = torch.abs(box_pos[:, 2] - cart_pos[:, 2])
    
    stage1_current = box_cart_x_sep < box_to_cart_max_x
    stage1_current = torch.logical_and(stage1_current, box_cart_y_sep < box_to_cart_max_y)
    stage1_current = torch.logical_and(stage1_current, box_cart_z_sep < box_to_cart_max_z)
    
    # Check if stage1 just completed (newly achieved)
    stage1_newly_complete = torch.logical_and(
        stage1_current,
        torch.logical_not(_MULTI_STAGE_STATE["stage1_complete"])
    )
    
    # Update stage1 state (once complete, stays complete)
    _MULTI_STAGE_STATE["stage1_complete"] = torch.logical_or(
        _MULTI_STAGE_STATE["stage1_complete"], 
        stage1_current
    )
    
    # Print when stage1 is newly completed
    if stage1_newly_complete.any():
        print("✅ [Stage 1 Complete] Box successfully placed on cart!")
        print(f"   Box-Cart separation: X={box_cart_x_sep[0]:.3f}m, Y={box_cart_y_sep[0]:.3f}m, Z={box_cart_z_sep[0]:.3f}m")
    
    # --- Stage 2: Cart to Fixed Target Position ---
    cart_target_x_sep = torch.abs(cart_pos[:, 0] - target_pos[0, 0])
    cart_target_y_sep = torch.abs(cart_pos[:, 1] - target_pos[0, 1])
    cart_target_z_sep = torch.abs(cart_pos[:, 2] - target_pos[0, 2])
    
    stage2_current = cart_target_x_sep < cart_to_target_max_x
    stage2_current = torch.logical_and(stage2_current, cart_target_y_sep < cart_to_target_max_y)
    stage2_current = torch.logical_and(stage2_current, cart_target_z_sep < cart_to_target_max_z)
    
    # Check if stage2 just completed (newly achieved)
    stage2_newly_complete = torch.logical_and(
        stage2_current,
        torch.logical_not(_MULTI_STAGE_STATE["stage2_complete"])
    )
    
    # Update stage2 state (once complete, stays complete)
    _MULTI_STAGE_STATE["stage2_complete"] = torch.logical_or(
        _MULTI_STAGE_STATE["stage2_complete"],
        stage2_current
    )
    
    # Print when stage2 is newly completed
    if stage2_newly_complete.any():
        print("✅ [Stage 2 Complete] Cart successfully pushed to target position!")
        print(f"   Cart position: ({cart_pos[0, 0]:.3f}, {cart_pos[0, 1]:.3f}, {cart_pos[0, 2]:.3f})")
        print(f"   Target position: ({target_pos[0, 0]:.3f}, {target_pos[0, 1]:.3f}, {target_pos[0, 2]:.3f})")
        print(f"   Separation: X={cart_target_x_sep[0]:.3f}m, Y={cart_target_y_sep[0]:.3f}m, Z={cart_target_z_sep[0]:.3f}m")
    
    # --- Success: Both stages complete ---
    both_stages_complete = torch.logical_and(
        _MULTI_STAGE_STATE["stage1_complete"],
        _MULTI_STAGE_STATE["stage2_complete"]
    )
    
    # Print when both stages are complete (only print once per episode)
    if both_stages_complete.any():
        for env_idx in range(num_envs):
            if both_stages_complete[env_idx]:
                # Check if we've already printed for this env
                if 'success_printed_envs' not in _MULTI_STAGE_STATE:
                    _MULTI_STAGE_STATE['success_printed_envs'] = set()
                
                if env_idx not in _MULTI_STAGE_STATE['success_printed_envs']:
                    print(f"🎉 [MISSION COMPLETE - Env {env_idx}] Both stages finished! Episode will terminate and reset.")
                    _MULTI_STAGE_STATE['success_printed_envs'].add(env_idx)
                
                # DO NOT reset state here - state persists until manual reset
                # This allows success_step_count to accumulate in record script
    
    return both_stages_complete
