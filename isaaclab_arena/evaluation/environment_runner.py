# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import time
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    import gymnasium as gym


class RealTimeRateLimiter:
    """Limit a loop to one iteration per requested period."""

    def __init__(self, period: float) -> None:
        """Initialize the limiter for a period in seconds.

        Args:
            period: Minimum time between consecutive loop iterations.
        """
        self._period = period
        self._next_iteration_time = time.monotonic()

    def sleep(self) -> None:
        """Sleep until the next iteration should begin."""
        self._next_iteration_time += self._period
        current_time = time.monotonic()
        remaining_time = self._next_iteration_time - current_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)
        else:
            self._next_iteration_time = current_time


def use_kit_visualizer_by_default(args_cli: argparse.Namespace) -> None:
    """Select the Kit visualizer when the user did not select a visualizer."""
    visualizer_was_explicit = getattr(args_cli, "visualizer_explicit", False)
    if args_cli.visualizer is None and not visualizer_was_explicit and not args_cli.headless:
        args_cli.visualizer = ["kit"]


def assert_interactive_runner_args(args_cli: argparse.Namespace) -> None:
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


def assert_gui_environment() -> None:
    """Check that environment variables do not force Isaac Sim into headless mode."""
    assert os.environ.get("HEADLESS", "0") == "0", "environment_runner requires the Kit GUI; unset HEADLESS"


def disable_timeout_terminations(env_cfg) -> list[str]:
    """Disable timeout termination terms and return their names."""
    terminations_cfg = env_cfg.terminations
    if terminations_cfg is None:
        return []

    disabled_term_names: list[str] = []
    if hasattr(terminations_cfg, "time_out") and terminations_cfg.time_out is not None:
        terminations_cfg.time_out = None
        disabled_term_names.append("time_out")

    for term_name, term_cfg in vars(terminations_cfg).items():
        if term_cfg is not None and getattr(term_cfg, "time_out", False):
            setattr(terminations_cfg, term_name, None)
            if term_name not in disabled_term_names:
                disabled_term_names.append(term_name)

    return disabled_term_names


def get_idle_actions(env: gym.Env, env_cfg) -> torch.Tensor:
    """Return the configured idle action or a zero action for the environment."""
    device = torch.device(env.unwrapped.device)
    configured_idle_action = getattr(env_cfg, "idle_action", None)
    if configured_idle_action is not None:
        return configured_idle_action.repeat(env.unwrapped.num_envs, 1).to(device)
    return torch.zeros(env.action_space.shape, device=device)


def get_fired_termination_terms(env: gym.Env) -> dict[str, list[int]]:
    """Return environment IDs for every termination term that fired this step."""
    termination_manager = env.unwrapped.termination_manager
    fired_terms: dict[str, list[int]] = {}
    for term_name in termination_manager.active_terms:
        term_env_ids = termination_manager.get_term(term_name).nonzero(as_tuple=False).flatten().tolist()
        if term_env_ids:
            fired_terms[term_name] = term_env_ids
    return fired_terms


def log_fired_termination_terms(env: gym.Env, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
    """Print the named terms responsible for an automatic environment reset."""
    if not (terminated | truncated).any().item():
        return

    fired_terms = get_fired_termination_terms(env)
    for term_name, env_ids in fired_terms.items():
        print(
            f"[environment_runner] Termination '{term_name}' fired for env IDs {env_ids}; environment reset.",
            flush=True,
        )


def enable_mouse_interaction() -> None:
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


def run_environment(
    simulation_app: SimulationAppContext,
    env: gym.Env,
    env_cfg,
    rate_limiter: RealTimeRateLimiter | None = None,
) -> None:
    """Reset once, then run one environment with idle actions until Kit closes."""
    env.reset()
    idle_actions = get_idle_actions(env, env_cfg)
    if rate_limiter is None:
        rate_limiter = RealTimeRateLimiter(env.unwrapped.step_dt)

    print(
        "[environment_runner] Environment running. Hold Shift, then left-drag a physics object.",
        flush=True,
    )
    print("[environment_runner] Close the Kit window or press Ctrl-C to exit.", flush=True)

    try:
        with torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                _, _, terminated, truncated, _ = env.step(idle_actions)
                log_fired_termination_terms(env, terminated, truncated)
                rate_limiter.sleep()
    except KeyboardInterrupt:
        print("\n[environment_runner] Exiting.", flush=True)


def main() -> None:
    """Launch and continuously run one interactive Arena environment."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.set_defaults(device="cpu")
    args_parser.allow_abbrev = False
    app_args, _ = args_parser.parse_known_args()
    use_kit_visualizer_by_default(app_args)
    assert_interactive_runner_args(app_args)
    assert_gui_environment()
    print("[environment_runner] Using CPU physics for interactive viewport manipulation.", flush=True)

    with SimulationAppContext(app_args) as simulation_app:
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        use_kit_visualizer_by_default(args_cli)
        assert_interactive_runner_args(args_cli)
        assert_hydra_overrides(hydra_overrides, args_parser)

        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
        disabled_term_names = disable_timeout_terminations(env_cfg)
        env_cfg.sim.enable_scene_query_support = True
        env_cfg.recorders = {}
        env_cfg.episode_recorders = {}
        if disabled_term_names:
            print(
                f"[environment_runner] Disabled timeout terms: {', '.join(disabled_term_names)}.",
                flush=True,
            )

        env = arena_builder.make_registered(env_cfg, env_kwargs)
        try:
            enable_mouse_interaction()
            run_environment(simulation_app, env, env_cfg)
        finally:
            env.close()


if __name__ == "__main__":
    main()
