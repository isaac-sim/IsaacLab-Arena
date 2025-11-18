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

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.tasks.terminations_orca import multi_stage_with_fixed_destination
from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask, EventsCfg as ParentEventsCfg
from isaaclab_arena.terms.events import set_object_pose

class G1OrcaTask(G1LocomanipPickAndPlaceTask):
    def get_mimic_env_cfg(self, embodiment_name: str):
        """Override parent to return ORCA-specific mimic environment configuration."""
        # Get parent's env config
        env_cfg = super().get_mimic_env_cfg(embodiment_name)
        
        # Override datagen config for ORCA task
        env_cfg.datagen_config.name = "g1_orca_task_D0"
        
        # Clear parent's subtask configs and define ORCA-specific ones
        env_cfg.subtask_configs = {}
        
        # Right hand subtasks: before grasp, after release, hand on cart
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="brown_box",
                subtask_term_signal="right_before_grasp_box",  # Before grasping box
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="brown_box",
                subtask_term_signal="right_after_release_box",  # After releasing box
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - hand relative to cart
                subtask_term_signal="right_hand_on_cart",  # Hand on cart
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        # Last subtask (final state, no need to annotate)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - pushing cart
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        env_cfg.subtask_configs["right"] = subtask_configs
        
        # Left hand subtasks: before grasp, after release, hand on cart
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="brown_box",
                subtask_term_signal="left_before_grasp_box",  # Before grasping box
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="brown_box",
                subtask_term_signal="left_after_release_box",  # After releasing box
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - hand relative to cart
                subtask_term_signal="left_hand_on_cart",  # Hand on cart
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        # Last subtask (final state, no need to annotate)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - pushing cart
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        env_cfg.subtask_configs["left"] = subtask_configs
        
        # Body subtasks: face shelf, face cart, reach destination
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="brown_box",
                subtask_term_signal="body_face_shelf",  # Face the shelf/table
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - navigating to cart
                subtask_term_signal="body_face_cart",  # Face the cart
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - pushing cart to destination
                subtask_term_signal="body_reach_destination",  # Reach destination
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        # Last subtask (final state, no need to annotate)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="orca_cart",  # Changed to orca_cart - final state with cart
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        env_cfg.subtask_configs["body"] = subtask_configs
        
        return env_cfg
    
    def get_events_cfg(self):
        """Override parent to add orca_cart reset event."""
        # Create custom EventsCfg for ORCA task
        events_cfg = OrcaEventsCfg(
            pick_up_object=self.pick_up_object,
            destination_cart=self.destination_bin  # Add cart reset
        )
        return events_cfg
    
    def get_termination_cfg(self):
        # Two-stage success: 
        # Stage 1: Box on cart
        # Stage 2: Cart to target position (surgical bed)
        success = TerminationTermCfg(
            func=multi_stage_with_fixed_destination,
            params={
                # Stage 1: Box on cart
                "box_cfg": SceneEntityCfg(self.pick_up_object.name),
                "cart_cfg": SceneEntityCfg(self.destination_bin.name),
                "box_to_cart_max_x": 0.30,  # 30cm tolerance in X
                "box_to_cart_max_y": 0.20,  # 20cm tolerance in Y
                "box_to_cart_max_z": 0.92,  # 25cm tolerance in Z (box above cart)
                # Stage 2: Cart to fixed target position (surgical bed)
                "target_position_x": 0.137,
                "target_position_y": -4.5,
                "target_position_z": -0.7875,
                "cart_to_target_max_x": 0.75,  # 50cm tolerance in X
                "cart_to_target_max_y": 0.75,  # 50cm tolerance in Y (increased for replay stability)
                "cart_to_target_max_z": 0.30,  # 30cm tolerance in Z
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": -0.6,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class OrcaEventsCfg(ParentEventsCfg):
    """Event configuration for ORCA task - adds cart reset to parent's box reset."""

    reset_destination_cart_pose: EventTermCfg = MISSING

    def __init__(self, pick_up_object, destination_cart):
        """Initialize reset events for both brown_box and orca_cart.
        
        Args:
            pick_up_object: The box to pick up (brown_box)
            destination_cart: The cart to push (orca_cart)
        """
        # Inherit parent class to handle box reset automatically
        super().__init__(pick_up_object)
        
        # Add cart reset event
        cart_initial_pose = destination_cart.get_initial_pose()
        if cart_initial_pose is not None:
            self.reset_destination_cart_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": cart_initial_pose,
                    "asset_cfg": SceneEntityCfg(destination_cart.name),
                },
            )
        else:
            print(f"Destination cart {destination_cart.name} has no initial pose.")
            self.reset_destination_cart_pose = None
