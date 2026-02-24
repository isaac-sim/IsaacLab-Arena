# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Demo script that records a video of a policy running while cycling through HDR backgrounds.

The HDR environment map on the dome light is swapped every ``--steps_per_hdr`` simulation
steps, producing a single continuous video where the background flashes through all
registered HDR textures.

python isaaclab_arena/scripts/demo/hdr_flash_demo.py     --num_steps 660     --steps_per_hdr 60     --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml     droid_pick_and_place_srl

Example usage::

    python isaaclab_arena/scripts/demo/hdr_flash_demo.py \
        --enable_cameras \
        --num_steps 660 \
        --steps_per_hdr 60 \
        --policy_type zero_action \
        droid_pick_and_place_srl

    python isaaclab_arena/scripts/demo/hdr_flash_demo.py \
        --enable_cameras \
        --num_steps 660 \
        --steps_per_hdr 60 \
        --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
        droid_pick_and_place_srl
"""

"""Launch Isaac Sim Simulator first."""

import gymnasium as gym
import os

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments

# Build the argument parser (environment subcommands and policy-specific args
# are added later inside main() to avoid argparse conflicts with unknown flags).
parser = get_isaaclab_arena_cli_parser()

# Demo-specific arguments
parser.add_argument(
    "--steps_per_hdr",
    type=int,
    default=60,
    help="Number of simulation steps to show each HDR before switching. At 30 Hz this is ~2 seconds.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=os.path.join("logs", "demo", "hdr_flash_demo"),
    help="Directory to save the recorded video.",
)

# Policy runner arguments (--policy_type, --num_steps, --num_episodes)
add_policy_runner_arguments(parser)

# Pre-parse to get known args before simulation starts
args_cli, unknown = parser.parse_known_args()

# Cameras are required for video recording
args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# Launch the simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import tqdm

import isaaclab_tasks  # noqa: F401
from pxr import UsdLux

from isaaclab_arena.assets.asset_registry import HDRImageRegistry
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.utils.usd_helpers import get_all_prims
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def _get_dome_light_prim(env):
    """Find the DomeLight prim in the stage."""
    all_prims = get_all_prims(env.unwrapped.scene.stage)
    light_prims = [prim for prim in all_prims if prim.IsA(UsdLux.DomeLight)]
    assert len(light_prims) > 0, "No DomeLight prim found in stage"
    return light_prims[0]


def _collect_hdr_texture_paths() -> list[tuple[str, str]]:
    """Return (name, texture_file) pairs for every registered HDR."""
    registry = HDRImageRegistry()
    keys = registry.get_all_keys()
    result = []
    for key in keys:
        hdr_cls = registry.get_hdr_by_name(key)
        hdr = hdr_cls()
        result.append((hdr.name, hdr.texture_file))
    return result


def main() -> None:
    # Resolve the policy class, add environment subcommands + policy args, then do full parse.
    # This must happen after AppLauncher so that policy-specific flags (e.g. --policy_config_yaml_path)
    # don't get consumed by the environment subparser.
    policy_cls = get_policy_cls(args_cli.policy_type)
    get_isaaclab_arena_environments_cli_parser(parser)
    policy_cls.add_args_to_parser(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    # Build the Arena environment
    arena_builder = get_arena_builder_from_cli(args)
    env_name, env_cfg = arena_builder.build_registered()

    # Create with render_mode for video capture
    env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array")

    # Determine total steps
    num_steps = args.num_steps
    if num_steps is None:
        policy = policy_cls.from_args(args)
        if policy.has_length():
            num_steps = policy.length()
        else:
            raise ValueError("--num_steps is required (policy does not define a length)")
    else:
        policy = policy_cls.from_args(args)

    # Wrap with video recorder
    video_dir = os.path.abspath(args.video_dir)
    os.makedirs(video_dir, exist_ok=True)
    video_kwargs = {
        "video_folder": video_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": num_steps,
        "disable_logger": True,
    }
    print(f"[HDR Flash Demo] Recording video to: {video_dir}")
    print(f"[HDR Flash Demo] Total steps: {num_steps}, steps per HDR: {args.steps_per_hdr}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Collect all HDR textures
    hdr_textures = _collect_hdr_texture_paths()
    num_hdrs = len(hdr_textures)
    print(f"[HDR Flash Demo] Cycling through {num_hdrs} HDR backgrounds:")
    for i, (name, _) in enumerate(hdr_textures):
        print(f"  [{i}] {name}")

    # Reset and get the dome light prim
    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(env.unwrapped.cfg.isaaclab_arena_env.task.get_task_description())

    dome_light_prim = _get_dome_light_prim(env)
    texture_attr = dome_light_prim.GetAttribute("inputs:texture:file")

    # Set the initial HDR
    hdr_idx = 0
    hdr_name, hdr_path = hdr_textures[hdr_idx]
    texture_attr.Set(hdr_path)
    print(f"[HDR Flash Demo] Step 0: switching to HDR '{hdr_name}'")

    # Main loop
    assert num_steps is not None
    pbar = tqdm.tqdm(total=num_steps, desc="HDR Flash Demo", unit="step")
    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)

            if terminated.any() or truncated.any():
                env_ids = (terminated | truncated).nonzero().flatten()
                policy.reset(env_ids=env_ids)

        # Swap HDR at the specified interval
        if (step + 1) % args.steps_per_hdr == 0:
            hdr_idx = (hdr_idx + 1) % num_hdrs
            hdr_name, hdr_path = hdr_textures[hdr_idx]
            texture_attr.Set(hdr_path)
            print(f"[HDR Flash Demo] Step {step + 1}: switching to HDR '{hdr_name}'")

        pbar.update(1)

    pbar.close()
    print(f"[HDR Flash Demo] Done. Video saved to: {video_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
