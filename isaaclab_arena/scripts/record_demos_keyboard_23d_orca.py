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


import contextlib
import gymnasium as gym
import os
import time
import torch

from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

# Add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful.",
)
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=0.1,
    help="Position sensitivity (meters per key press).",
)
parser.add_argument(
    "--rot_sensitivity",
    type=float,
    default=0.1,
    help="Rotation sensitivity (radians per key press).",
)
parser.add_argument(
    "--vel_sensitivity",
    type=float,
    default=0.1,
    help="Velocity sensitivity (m/s per key press).",
)
parser.add_argument(
    "--height_sensitivity",
    type=float,
    default=0.01,
    help="Height sensitivity (meters per key press).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# Add the example environments CLI args
add_example_environments_cli_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.log
import omni.ui as ui
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from isaaclab_arena.teleop_devices.keyboard_23d_adapter import KeyboardTo23DAdapter, KeyboardTo23DConfig
from isaaclab_arena.tasks.terminations_orca import reset_multi_stage_state


class RateLimiter:
    """Convenience class for enforcing rates in loops."""
    
    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.
        
        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)
    
    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.
        
        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        
        self.last_time = self.last_time + self.sleep_duration
        
        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            self.last_time = time.time()


def setup_output_directories() -> tuple[str, str]:
    """Setup output directories for saving demonstrations."""
    output_filepath = os.path.abspath(args_cli.dataset_file)
    output_dir = os.path.dirname(output_filepath)
    output_file_name = os.path.basename(output_filepath)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir, output_file_name


def create_environment_config(output_dir: str, output_file_name: str):
    """Create environment configuration with recording enabled."""
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()  # Fixed: correct order
    
    # Enable mimic environment for recording
    env_cfg.is_finite_horizon = False
    # Use environment's default episode_length_s (typically 100s or more)
    
    # Setup recording
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    
    # Extract success termination for manual checking (following record_demos.py pattern)
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None  # Disable auto-reset on success
        print(f"[INFO] Success condition extracted: {success_term.func.__name__}")
        print(f"[INFO] Success parameters: {success_term.params}")
    else:
        print("[WARNING] Success detection and auto-save will be disabled")
    
    # Disable timeout
    env_cfg.terminations.time_out = None
    
    # Keep observations as dictionary (not concatenated tensor)
    # This is CRITICAL for camera data to be recorded properly!
    env_cfg.observations.policy.concatenate_terms = False
    
    return env_cfg, env_name, success_term


def main() -> None:
    """Main function for recording demonstrations."""
    
    # Setup
    rate_limiter = RateLimiter(args_cli.step_hz)
    output_dir, output_file_name = setup_output_directories()
    
    # Create environment config (success termination already extracted and disabled)
    env_cfg, env_name, success_term = create_environment_config(output_dir, output_file_name)
    
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        print(f"[INFO] Environment created: {env_name}")
        print(f"[INFO] Action space shape: {env.action_space.shape}")
        
        # Verify 23D action space (check last dimension, not batch dimension)
        action_dim = env.action_space.shape[-1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
        if action_dim != 23:
            omni.log.error(f"Action space is {action_dim}D, expected 23D")
            omni.log.error("Please use G1 WBC embodiments: --embodiment g1_wbc_pink or g1_wbc_joint")
            env.close()
            simulation_app.close()
            return
        
        # Measure and print cart information (one-time measurement) - COMMENTED OUT
        # print("\n" + "="*80)
        # print("Scene Object Measurements:")
        # print("="*80)
        # if "orca_cart" in env.scene.rigid_objects:
        #     cart = env.scene.rigid_objects["orca_cart"]
        #     cart_pos = cart.data.root_pos_w[0].cpu().numpy()
        #     cart_pos_rel = (cart.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy()
        #     
        #     # Get bounding box from USD
        #     import omni.usd
        #     from pxr import UsdGeom, Usd
        #     stage = omni.usd.get_context().get_stage()
        #     cart_prim_path = f"/World/envs/env_0/orca_cart"
        #     cart_prim = stage.GetPrimAtPath(cart_prim_path)
        #     
        #     bbox_info = "Unable to obtain"
        #     height = 0.0
        #     try:
        #         if cart_prim.IsValid():
        #             bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        #             bbox = bbox_cache.ComputeWorldBound(cart_prim)
        #             bbox_range = bbox.GetRange()
        #             if bbox_range:
        #                 min_point = bbox_range.GetMin()
        #                 max_point = bbox_range.GetMax()
        #                 width = max_point[0] - min_point[0]
        #                 depth = max_point[1] - min_point[1]
        #                 height = max_point[2] - min_point[2]
        #                 bbox_info = f"W={width:.3f}m, D={depth:.3f}m, H={height:.3f}m ({height*100:.1f}cm)"
        #     except Exception as e:
        #         bbox_info = f"Failed: {e}"
        #     
        #     print(f"Cart (orca_cart):")
        #     print(f"   Bottom height: {cart_pos_rel[2]:.4f}m, Top height: ~{cart_pos_rel[2] + (height if 'height' in locals() else 0):.4f}m")
        # print("="*80 + "\n")
            
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return
    
    # Recording state
    current_demo_count = 0
    success_step_count = 0
    should_reset = False
    
    # Callbacks
    def reset_recording():
        nonlocal should_reset
        should_reset = True
    
    # Create keyboard adapter
    config = KeyboardTo23DConfig(
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
        vel_sensitivity=args_cli.vel_sensitivity,
        height_sensitivity=args_cli.height_sensitivity,
    )
    
    teleop_interface = KeyboardTo23DAdapter(cfg=config, sim_device=args_cli.device)
    teleop_interface.add_callback("ENTER", reset_recording)
    
    # Setup UI
    label_text = f"Recording demonstration {current_demo_count + 1}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'}"
    instruction_display = InstructionDisplay(xr=False)
    window = EmptyWindow(env, "Recording Status")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
        instruction_display.set_labels(subtask_label, demo_label)
    
    # Reset and start
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    reset_multi_stage_state()  # Clear multi-stage task state
    
    print("\n" + "=" * 60)
    print(f"Recording started! Target: {args_cli.num_demos if args_cli.num_demos > 0 else 'unlimited'} successful demos")
    print("Complete the task successfully to save the demonstration")
    print("Press [ENTER] to reset environment")
    print("=" * 60 + "\n")
    
    # Main loop
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get keyboard action (23D)
                action = teleop_interface.advance()
                
                # Apply to all environments
                actions = action.unsqueeze(0).repeat(env.num_envs, 1)
                env.step(actions)
                
                # Check if environment was auto-reset due to termination
                if env.reset_buf[0]:
                    # Environment was reset, log diagnostic info
                    episode_length = int(env.episode_length_buf[0]) if env.episode_length_buf[0] > 0 else 0
                    
                    print(f"\n{'='*70}")
                    print(f" AUTO-RESET DETECTED!")
                    print(f"{'='*70}")
                    print(f"Episode length: {episode_length} steps")
                    
                    # Try to get box position if success_term exists
                    if success_term is not None:
                        try:
                            # For multi-stage tasks, show box position
                            if "box_cfg" in success_term.params:
                                object_pos = env.scene[success_term.params["box_cfg"].name].data.root_pos_w - env.scene.env_origins
                            elif "object_cfg" in success_term.params:
                                object_pos = env.scene[success_term.params["object_cfg"].name].data.root_pos_w - env.scene.env_origins
                            else:
                                raise KeyError("No object config found")
                            object_z = float(object_pos[0, 2])
                            print(f"Box Z-height at reset: {object_z:.3f}m")
                        except:
                            pass
                    
                    if env.reset_terminated[0]:
                        print(f"\n Termination Reason:")
                        # Check if it's object dropped termination
                        if success_term is not None:
                            try:
                                # For multi-stage tasks, check box position
                                if "box_cfg" in success_term.params:
                                    object_pos = env.scene[success_term.params["box_cfg"].name].data.root_pos_w - env.scene.env_origins
                                elif "object_cfg" in success_term.params:
                                    object_pos = env.scene[success_term.params["object_cfg"].name].data.root_pos_w - env.scene.env_origins
                                else:
                                    raise KeyError("No object config found")
                                object_z = float(object_pos[0, 2])
                                if object_z < -0.6:
                                    print(f"Box dropped to ground! (Z={object_z:.3f}m < -0.6m)")
                                else:
                                    print(f"Other termination condition")
                            except:
                                print(f"Termination condition triggered")
                        else:
                            print(f"Termination condition triggered")
                    elif env.reset_time_outs[0]:
                        print(f"\n  Episode Timeout Reached")
                        max_episode_length = env.max_episode_length
                        print(f"   Max episode length: {max_episode_length} steps")
                    
                    print(f"{'='*70}\n")
                    
                    # Reset counters
                    success_step_count = 0
                    teleop_interface.reset()
                    continue
                
                # Check success condition
                if success_term is not None:
                    # Check success (lightweight boolean check)
                    is_success = bool(success_term.func(env, **success_term.params)[0])
                    
                    if is_success:
                        success_step_count += 1
                        
                        # Only calculate distances for printing (reduce frequency)
                        if success_step_count % 20 == 1:
                            # Check if this is a multi-stage task
                            if "box_cfg" in success_term.params and "cart_cfg" in success_term.params:
                                # Multi-stage: just show progress count
                                print(f"SUCCESS PROGRESS: {success_step_count}/{args_cli.num_success_steps} steps | "
                                      f"Both stages complete - holding position...")
                            elif "object_cfg" in success_term.params and "target_object_cfg" in success_term.params:
                                # Single-stage: show distance
                                object_pos = env.scene[success_term.params["object_cfg"].name].data.root_pos_w - env.scene.env_origins
                                target_pos = env.scene[success_term.params["target_object_cfg"].name].data.root_pos_w - env.scene.env_origins
                                x_sep = float(torch.abs(object_pos[0, 0] - target_pos[0, 0]))
                                y_sep = float(torch.abs(object_pos[0, 1] - target_pos[0, 1]))
                                z_sep = float(torch.abs(object_pos[0, 2] - target_pos[0, 2]))
                                print(f"SUCCESS PROGRESS: {success_step_count}/{args_cli.num_success_steps} steps | "
                                      f"Distance: X={x_sep:.3f}m Y={y_sep:.3f}m Z={z_sep:.3f}m")
                        
                        if success_step_count >= args_cli.num_success_steps:
                            # Mark episode as successful and export
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                            )
                            env.recorder_manager.export_episodes([0])
                            current_demo_count += 1
                            
                            print(f"\n{'='*60}")
                            print(f"✅ Demo {current_demo_count} saved successfully!")
                            print(f"{'='*60}\n")
                            
                            # Update UI
                            label_text = f"Recording demonstration {current_demo_count + 1}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'}"
                            instruction_display.show_demo(label_text)
                            
                            # Check if we've reached target
                            if args_cli.num_demos > 0 and current_demo_count >= args_cli.num_demos:
                                print(f"\n[INFO] Target reached: {current_demo_count} demonstrations recorded")
                                break
                            
                            # Auto reset for next demo
                            env.reset()
                            teleop_interface.reset()
                            reset_multi_stage_state()  # Clear multi-stage task state
                            success_step_count = 0
                    else:
                        # Reset counter when not in success region
                        success_step_count = 0
                        
                        # Print distance feedback every 200 frames (less frequent to avoid lag)
                        if env.episode_length_buf[0] % 200 == 0:
                            # Check if this is a multi-stage task (has box_cfg and cart_cfg)
                            if "box_cfg" in success_term.params and "cart_cfg" in success_term.params:
                                # Two-stage task: show both stage distances
                                box_pos = env.scene[success_term.params["box_cfg"].name].data.root_pos_w - env.scene.env_origins
                                cart_pos = env.scene[success_term.params["cart_cfg"].name].data.root_pos_w - env.scene.env_origins
                                
                                # Stage 1: Box to Cart
                                box_cart_x = float(torch.abs(box_pos[0, 0] - cart_pos[0, 0]))
                                box_cart_y = float(torch.abs(box_pos[0, 1] - cart_pos[0, 1]))
                                box_cart_z = float(torch.abs(box_pos[0, 2] - cart_pos[0, 2]))
                                max_x1 = success_term.params["box_to_cart_max_x"]
                                max_y1 = success_term.params["box_to_cart_max_y"]
                                max_z1 = success_term.params["box_to_cart_max_z"]
                                
                                # Stage 2: Cart to Target
                                target_x = success_term.params["target_position_x"]
                                target_y = success_term.params["target_position_y"]
                                target_z = success_term.params["target_position_z"]
                                cart_target_x = float(torch.abs(cart_pos[0, 0] - target_x))
                                cart_target_y = float(torch.abs(cart_pos[0, 1] - target_y))
                                cart_target_z = float(torch.abs(cart_pos[0, 2] - target_z))
                                max_x2 = success_term.params["cart_to_target_max_x"]
                                max_y2 = success_term.params["cart_to_target_max_y"]
                                max_z2 = success_term.params["cart_to_target_max_z"]
                                
                                print(f"📦 [Stage 1: Box→Cart] X={box_cart_x:.3f}/{max_x1:.2f}m  Y={box_cart_y:.3f}/{max_y1:.2f}m  Z={box_cart_z:.3f}/{max_z1:.2f}m")
                                print(f"🚚 [Stage 2: Cart→Target] X={cart_target_x:.3f}/{max_x2:.2f}m  Y={cart_target_y:.3f}/{max_y2:.2f}m  Z={cart_target_z:.3f}/{max_z2:.2f}m")
                            elif "object_cfg" in success_term.params and "target_object_cfg" in success_term.params:
                                # Single-stage task: original code
                                object_pos = env.scene[success_term.params["object_cfg"].name].data.root_pos_w - env.scene.env_origins
                                target_pos = env.scene[success_term.params["target_object_cfg"].name].data.root_pos_w - env.scene.env_origins
                                x_sep = float(torch.abs(object_pos[0, 0] - target_pos[0, 0]))
                                y_sep = float(torch.abs(object_pos[0, 1] - target_pos[0, 1]))
                                z_sep = float(torch.abs(object_pos[0, 2] - target_pos[0, 2]))
                                max_x = success_term.params["max_x_separation"]
                                max_y = success_term.params["max_y_separation"]
                                max_z = success_term.params["max_z_separation"]
                                print(f"📍 Distance: X={x_sep:.3f}/{max_x:.2f}m  Y={y_sep:.3f}/{max_y:.2f}m  Z={z_sep:.3f}/{max_z:.2f}m")
                
                # Handle manual reset
                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    reset_multi_stage_state()  # Clear multi-stage task state
                    success_step_count = 0
                    should_reset = False
                
                # Rate limiting
                rate_limiter.sleep(env)
                
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")
    
    # Cleanup
    env.close()
    print(f"\n{'='*60}")
    print(f"Recording completed!")
    print(f"Total successful demonstrations: {current_demo_count}")
    print(f"Saved to: {args_cli.dataset_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

