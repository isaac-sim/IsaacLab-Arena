# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Load an untouched Replicator kitchen and report background settlement.

Run inside the Arena development container:

    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py
    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py --record_camera_video
    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py --kitchen_source s3
    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py --kitchen_source datasets
    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py --kitchen_layout g_shape
    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py --steps 10
    /isaac-sim/python.sh isaaclab_arena/scripts/check_replicator_kitchen_settlement.py --sim_freq 60 --decimation 4

The script does not alter the background objects or robot placement. It resets the
environment, holds the robot at its reset joint pose, advances the untouched scene
for ``--steps`` environment steps, and reports every nested rigid background asset
that has not settled.
"""

from __future__ import annotations

import argparse
import torch
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.rate_limiter import RateLimiter
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir, wrap_env_for_video
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    import gymnasium as gym


ENV_SPEC = "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_l_shape_mustard_bowl.yaml"
DATASETS_KITCHEN_DIR = "/datasets/assets/kitchen"
KITCHEN_ENV_SPECS = {
    "g_shape": "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_g_shape_mustard_bowl.yaml",
    "l_shape": ENV_SPEC,
    "peninsula": "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_peninsula_mustard_bowl.yaml",
}
DATASETS_KITCHEN_LAYOUTS = set(KITCHEN_ENV_SPECS)
SETTLED_LINEAR_SPEED_M_S = 0.01
SETTLED_ANGULAR_SPEED_RAD_S = 0.1
LOG_PREFIX = "[replicator_kitchen_settlement]"


def _select_kitchen(env_spec: str, source: str) -> tuple[str, str]:
    """Point this process at the requested hosted or local-dataset kitchen."""
    from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR, ISAAC_STAGING_NUCLEUS_DIR
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    background_name = ArenaEnvGraphSpec.from_yaml(env_spec).background.registry_name
    prefix = "replicator_kitchen_"
    assert background_name.startswith(prefix), f"Expected a Replicator kitchen background, got '{background_name}'"
    layout = background_name.removeprefix(prefix)
    if source == "datasets":
        assert layout in DATASETS_KITCHEN_LAYOUTS, (
            f"No local dataset kitchen is configured for '{layout}'; available layouts: "
            f"{sorted(DATASETS_KITCHEN_LAYOUTS)}"
        )
        kitchen_path = f"{DATASETS_KITCHEN_DIR}/kitchen_{layout}.usda"
    else:
        relative_paths = {
            "s3": f"Environments/replicator_kitchen/kitchen_{layout}.usda",
            "ov": f"Arena/assets/background_library/replicator_kitchen/kitchen_{layout}.usda",
        }
        asset_roots = {"s3": ISAAC_STAGING_NUCLEUS_DIR, "ov": ARENA_NUCLEUS_DIR}
        kitchen_path = f"{asset_roots[source]}/{relative_paths[source]}"
    background_class = AssetRegistry().get_asset_by_name(background_name)
    background_class.usd_path = kitchen_path
    return background_name, kitchen_path


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse arguments for one Replicator kitchen environment."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of environment steps to run before measuring settlement; defaults to 10.",
    )
    args_parser.add_argument(
        "--record_camera_video",
        action="store_true",
        default=False,
        help="Record one mp4 per camera in obs['camera_obs'], matching Arena policy-runner output.",
    )
    args_parser.add_argument(
        "--output_base_dir",
        type=str,
        default="outputs",
        help="Base directory for camera videos; a timestamped run directory is added.",
    )
    args_parser.add_argument(
        "--kitchen_source",
        choices=("s3", "ov", "datasets"),
        default="ov",
        help=f"Kitchen asset source: public Sim staging S3, legacy OV-hosted Arena, or local {DATASETS_KITCHEN_DIR}.",
    )
    args_parser.add_argument(
        "--kitchen_layout",
        choices=tuple(KITCHEN_ENV_SPECS),
        default=None,
        help="Select a Replicator mustard-bowl kitchen layout; overrides --env_spec when provided.",
    )
    args_parser.add_argument(
        "--sim_freq",
        "--sim_frequency",
        dest="sim_frequency_hz",
        type=float,
        default=None,
        help="Physics simulation frequency in Hz; defaults to the selected environment configuration.",
    )
    args_parser.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="Physics steps per environment/control step; defaults to the environment configuration.",
    )
    args_parser.set_defaults(env_spec=ENV_SPEC, num_envs=1, visualizer=["kit"])
    args_parser.allow_abbrev = False
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
    args_cli, hydra_overrides = args_parser.parse_known_args()
    assert_hydra_overrides(hydra_overrides, args_parser)
    if args_cli.kitchen_layout is not None:
        args_cli.env_spec = KITCHEN_ENV_SPECS[args_cli.kitchen_layout]
    assert args_cli.num_envs == 1, "The settlement probe targets env_0; use --num_envs 1"
    assert not args_cli.distributed, "This debug script does not support distributed execution"
    assert args_cli.steps > 0, "--steps must be positive"
    assert args_cli.sim_frequency_hz is None or args_cli.sim_frequency_hz > 0.0, "--sim_freq must be positive"
    assert args_cli.decimation is None or args_cli.decimation > 0, "--decimation must be positive"
    if args_cli.record_camera_video:
        args_cli.enable_cameras = True
    return args_cli, hydra_overrides


def _find_background_rigid_views(env: gym.Env, background_name: str) -> tuple[list[tuple[str, Any]], list[str]]:
    """Return usable rigid-body views and paths unavailable to the tensor backend."""
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.usd_prim_tree import find_nested_physics_roots

    base_env = env.unwrapped
    physics_view = base_env.sim.physics_manager.get_physics_sim_view()
    background_path = f"/World/envs/env_0/{background_name}"
    background_prim = base_env.scene.stage.GetPrimAtPath(background_path)
    assert background_prim.IsValid(), f"Background prim does not exist at '{background_path}'"
    rigid_views = []
    unavailable_paths = []
    for rigid_body_path, object_type in sorted(find_nested_physics_roots(background_prim).items()):
        if object_type != ObjectType.RIGID:
            continue
        rigid_body_view = physics_view.create_rigid_body_view(rigid_body_path)
        try:
            rigid_body_count = rigid_body_view.count
        except AttributeError:
            rigid_body_count = 0
        if rigid_body_count != 1:
            print(
                f"{LOG_PREFIX} WARNING: PhysX tensor view unavailable for "
                f"{rigid_body_path}; this asset will be reported as unmeasured.",
                flush=True,
            )
            unavailable_paths.append(rigid_body_path)
            continue
        rigid_views.append((rigid_body_path, rigid_body_view))
    return rigid_views, unavailable_paths


def _report_settlement(rigid_views: list[tuple[str, Any]], unavailable_paths: list[str]) -> None:
    """Print final speed and settlement status for every nested rigid background asset."""
    import warp as wp

    results = []
    for rigid_path, rigid_view in rigid_views:
        velocity = wp.to_torch(rigid_view.get_velocities())[0]
        linear_speed = torch.linalg.vector_norm(velocity[:3]).item()
        angular_speed = torch.linalg.vector_norm(velocity[3:]).item()
        settled = linear_speed < SETTLED_LINEAR_SPEED_M_S and angular_speed < SETTLED_ANGULAR_SPEED_RAD_S
        results.append((rigid_path, linear_speed, angular_speed, settled))

    settled_count = sum(settled for _, _, _, settled in results)
    print(
        f"{LOG_PREFIX} Settlement: {settled_count}/{len(results)} nested rigid assets "
        f"below {SETTLED_LINEAR_SPEED_M_S:g} m/s linear and {SETTLED_ANGULAR_SPEED_RAD_S:g} rad/s angular.",
        flush=True,
    )
    for rigid_path, linear_speed, angular_speed, settled in results:
        if not settled:
            print(
                f"{LOG_PREFIX} MOVING {rigid_path}: linear={linear_speed:.6f} m/s angular={angular_speed:.6f} rad/s",
                flush=True,
            )
    if unavailable_paths:
        print(
            f"{LOG_PREFIX} Unmeasured tensor views: {len(unavailable_paths)}.",
            flush=True,
        )
        for rigid_path in unavailable_paths:
            print(f"{LOG_PREFIX} UNMEASURED {rigid_path}", flush=True)


def _get_robot_hold_action(env: gym.Env) -> torch.Tensor:
    """Return an action that holds the DROID arm at its current reset pose."""
    base_env = env.unwrapped
    assert base_env.action_manager.active_terms == [
        "arm_action",
        "gripper_action",
    ], f"Expected DROID arm and gripper actions, got {base_env.action_manager.active_terms}"
    assert base_env.action_manager.action_term_dim == [
        7,
        1,
    ], f"Expected DROID action dimensions [7, 1], got {base_env.action_manager.action_term_dim}"
    robot = base_env.scene["robot"]
    arm_joint_ids, arm_joint_names = robot.find_joints(["panda_joint.*"], preserve_order=True)
    assert arm_joint_names == [
        f"panda_joint{joint_index}" for joint_index in range(1, 8)
    ], f"Unexpected DROID arm joint order: {arm_joint_names}"
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action[:, :7] = robot.data.joint_pos.torch[:, arm_joint_ids]
    return action


def run_environment(
    simulation_app: SimulationAppContext,
    env: gym.Env,
    steps: int,
    *,
    background_name: str,
) -> None:
    """Reset the untouched scene, advance it, and report background settlement."""
    env.reset()
    rigid_views, unavailable_paths = _find_background_rigid_views(env, background_name)
    robot_hold_action = _get_robot_hold_action(env)
    rate_limiter = RateLimiter(period_seconds=env.unwrapped.step_dt)

    print(
        f"{LOG_PREFIX} Scene left untouched; holding the robot reset pose for {steps} steps.",
        flush=True,
    )

    try:
        with torch.inference_mode():
            step_count = 0
            while simulation_app.is_running() and not simulation_app.is_exiting() and step_count < steps:
                env.step(robot_hold_action)
                rate_limiter.sleep()
                step_count += 1
    except KeyboardInterrupt:
        print(f"\n{LOG_PREFIX} Exiting.", flush=True)
    finally:
        _report_settlement(rigid_views, unavailable_paths)


def main() -> None:
    """Launch an untouched Replicator kitchen and measure background settlement."""
    args_cli, hydra_overrides = _parse_args()
    with SimulationAppContext(args_cli) as simulation_app:
        background_name, kitchen_path = _select_kitchen(args_cli.env_spec, args_cli.kitchen_source)
        print(
            f"{LOG_PREFIX} Loading {args_cli.kitchen_source} kitchen: {kitchen_path}",
            flush=True,
        )
        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
        if args_cli.sim_frequency_hz is not None:
            env_cfg.sim.dt = 1.0 / args_cli.sim_frequency_hz
        if args_cli.decimation is not None:
            env_cfg.decimation = args_cli.decimation
        physics_frequency_hz = 1.0 / env_cfg.sim.dt
        control_frequency_hz = physics_frequency_hz / env_cfg.decimation
        print(
            f"{LOG_PREFIX} Physics frequency: {physics_frequency_hz:g} Hz; "
            f"control frequency: {control_frequency_hz:g} Hz (decimation={env_cfg.decimation}).",
            flush=True,
        )
        video_cfg = VideoRecordingCfg(
            record_camera_video=args_cli.record_camera_video,
            video_base_dir=timestamped_run_dir(args_cli.output_base_dir),
            save_partial_camera_video=True,
        )
        env = arena_builder.make_registered(env_cfg, env_kwargs, render_mode=video_cfg.render_mode)
        env = wrap_env_for_video(env, video_cfg, num_steps=args_cli.steps, num_episodes=None)
        try:
            run_environment(
                simulation_app,
                env,
                steps=args_cli.steps,
                background_name=background_name,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
