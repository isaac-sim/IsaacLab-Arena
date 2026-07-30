# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Load a registered Arena environment or an environment graph spec and continuously step it in Kit. Use it to
physically manipulate objects with Shift + left-drag, inspect their placement and interactions, and verify
termination-triggered resets.

Run a registered environment:

    python isaaclab_arena/scripts/environment_runner.py \
        pick_and_place_maple_table \
        --embodiment droid_rel_joint_pos \
        --pick_up_object banana_ycb_robolab \
        --destination_location bowl_ycb_robolab

Run an environment graph spec:

    python isaaclab_arena/scripts/environment_runner.py \
        --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml
"""

from __future__ import annotations

import argparse
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.rate_limiter import RateLimiter
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    import gymnasium as gym


def _assert_interactive_runner_args(args_cli: argparse.Namespace) -> None:
    """Check that command-line arguments describe one interactive Kit environment."""
    assert not args_cli.headless, "environment_runner requires the Kit GUI; remove --headless"
    assert (
        args_cli.visualizer is not None and "kit" in args_cli.visualizer
    ), "environment_runner requires the Kit GUI; use --viz kit"
    assert args_cli.num_envs == 1, "environment_runner supports exactly one environment"
    assert not args_cli.distributed, "environment_runner does not support distributed execution"
    assert args_cli.presets != "newton", "environment_runner mouse interaction currently requires PhysX"
    assert not args_cli.list_variations, "environment_runner does not support --list_variations"
    assert args_cli.device == "cpu", "environment_runner mouse interaction requires CPU PhysX; use --device cpu"


def _parse_interactive_runner_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse and validate arguments for interactive environment inspection."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.set_defaults(device="cpu", visualizer=["kit"], disable_fabric=True)
    args_parser.allow_abbrev = False
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)

    args_cli, hydra_overrides = args_parser.parse_known_args()
    assert_hydra_overrides(hydra_overrides, args_parser)
    _assert_interactive_runner_args(args_cli)
    return args_cli, hydra_overrides


def _enable_mouse_interaction() -> None:
    """Enable Shift-drag physics interaction using a D6 joint grab."""
    import carb
    import omni.kit.app
    import omni.physx.bindings._physx as physx_bindings
    import omni.usd

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.physx.ui", True)

    mouse_interaction_boolean_settings = {
        physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED: True,
        physx_bindings.SETTING_MOUSE_GRAB: True,
        physx_bindings.SETTING_MOUSE_GRAB_WITH_FORCE: False,
        physx_bindings.SETTING_MOUSE_GRAB_IGNORE_INVISBLE: False,
    }
    settings = carb.settings.get_settings()
    for setting_path, setting_value in mouse_interaction_boolean_settings.items():
        settings.set_bool(setting_path, setting_value)

    stage = omni.usd.get_context().get_stage()
    assert stage is not None, "Cannot enable mouse interaction without an open USD stage"
    custom_layer_data = dict(stage.GetRootLayer().customLayerData)
    physics_settings = dict(custom_layer_data.get("physicsSettings", {}))
    physics_settings.update(mouse_interaction_boolean_settings)
    custom_layer_data["physicsSettings"] = physics_settings
    stage.GetRootLayer().customLayerData = custom_layer_data


def _create_interactive_environment(
    args_cli: argparse.Namespace,
    hydra_overrides: list[str],
) -> gym.Env:
    """Create an Arena environment configured for interactive manipulation."""
    arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
    env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
    # Enable mouse picking without recording the interactive session.
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    return arena_builder.make_registered(env_cfg, env_kwargs)


def run_environment(
    simulation_app: SimulationAppContext,
    env: gym.Env,
) -> None:
    """Reset once, then run one environment with zero actions until Kit closes."""
    env.reset()
    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    rate_limiter = RateLimiter(period_seconds=env.unwrapped.step_dt)

    print(
        "[environment_runner] Environment running. Hold Shift, then left-drag a physics object.",
        flush=True,
    )
    print("[environment_runner] Close the Kit window or press Ctrl-C to exit.", flush=True)

    try:
        with torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                env.step(zero_actions)
                rate_limiter.sleep()
    except KeyboardInterrupt:
        print("\n[environment_runner] Exiting.", flush=True)


def main() -> None:
    """Launch and continuously run one interactive Arena environment."""
    args_cli, hydra_overrides = _parse_interactive_runner_args()
    print("[environment_runner] Using CPU physics for interactive viewport manipulation.", flush=True)

    with SimulationAppContext(args_cli) as simulation_app:
        env = _create_interactive_environment(args_cli, hydra_overrides)
        try:
            _enable_mouse_interaction()
            run_environment(simulation_app, env)
        finally:
            env.close()


if __name__ == "__main__":
    main()
