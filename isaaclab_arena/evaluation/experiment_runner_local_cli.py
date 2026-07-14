# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Parse the Isaac Lab application arguments for local typed evaluation."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides


def parse_local_experiment_runner_args(cli_args: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse AppLauncher arguments and return trailing Experiment overrides."""
    parser = argparse.ArgumentParser(description="Run a typed Arena Experiment locally.")
    parser.add_argument(
        "--list_variations",
        action="store_true",
        help="Print configurable environment variations for each Run and exit.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.allow_abbrev = False
    app_launcher_args, experiment_overrides = parser.parse_known_args(cli_args)
    assert_hydra_overrides(experiment_overrides, parser)
    return app_launcher_args, experiment_overrides
