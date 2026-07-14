# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Select local execution or OSMO submission for an Experiment Runner configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentRunnerRoute:
    """Route one Experiment Runner configuration to local execution or OSMO."""

    config_path: str
    """Path to the typed Experiment Runner YAML."""

    local: bool
    """Whether to execute locally instead of submitting to OSMO."""

    osmo_config_path: str | None
    """Optional path to an OSMO workflow YAML, used only for submission."""


def should_show_experiment_runner_help(cli_args: list[str]) -> bool:
    """Return whether arguments request help for the typed Experiment Runner route."""
    requests_help = "--help" in cli_args or "-h" in cli_args
    typed_route_options = {"--config", "--local", "--osmo-config"}
    uses_typed_route_option = any(
        arg in typed_route_options or arg.startswith("--config=") or arg.startswith("--osmo-config=")
        for arg in cli_args
    )
    return requests_help and (len(cli_args) == 1 or uses_typed_route_option)


def print_experiment_runner_help() -> None:
    """Print the primary typed Experiment Runner interface."""
    parser = argparse.ArgumentParser(
        description="Run a YAML-configured Arena Experiment locally or submit it to OSMO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  Submit the bundled OpenPI example to OSMO using the default infrastructure config:
    /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \\
      --config isaaclab_arena_environments/evaluation_configs/openpi.yaml

  Run it locally against an already-running OpenPI server:
    /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \\
      --config isaaclab_arena_environments/evaluation_configs/openpi.yaml \\
      --local --enable_cameras

With --local, Isaac Lab AppLauncher arguments and trailing Hydra KEY=VALUE Experiment overrides
are accepted. Without --config, experiment_runner.py uses its deprecated legacy interface.""",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to an Experiment Runner YAML configuration.",
    )
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to OSMO.")
    parser.add_argument(
        "--osmo-config",
        metavar="PATH",
        help="Optional OSMO infrastructure YAML override; omit it to use osmo/config/arena_experiment_workflow.yaml.",
    )
    parser.print_help()


def uses_experiment_runner_cfg(cli_args: list[str]) -> bool:
    """Return whether arguments select the typed Experiment Runner route."""
    return any(arg == "--config" or arg.startswith("--config=") for arg in cli_args)


def parse_experiment_runner_route(cli_args: list[str]) -> tuple[ExperimentRunnerRoute, list[str]]:
    """Parse routing arguments and return arguments owned by the selected backend."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, help="Path to an Experiment Runner YAML configuration.")
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to OSMO.")
    parser.add_argument("--osmo-config", help="Path to an OSMO workflow YAML config.")
    route_args, backend_args = parser.parse_known_args(cli_args)

    if route_args.local and route_args.osmo_config is not None:
        parser.error("--osmo-config cannot be used with --local")

    return (
        ExperimentRunnerRoute(
            config_path=route_args.config,
            local=route_args.local,
            osmo_config_path=route_args.osmo_config,
        ),
        backend_args,
    )
