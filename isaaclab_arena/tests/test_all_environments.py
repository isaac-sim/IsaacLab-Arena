# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test registered environments that use the generic runtime backend.

Auto-discovers environments from the EnvironmentRegistry and runs each generic
environment for a few steps with the zero_action policy in one subprocess.
Custom-backend environments are covered by dedicated tests in their required
physics phase.
"""

import argparse

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.tests.test_experiment_runner import run_experiment_runner, write_jobs_config_to_file
from isaaclab_arena_environments.cli import add_environment_cli_args, ensure_environments_registered

NUM_STEPS = 2
HEADLESS = True


# This is a list of argument overrides for specific environments.
# Try to minimize the number of overrides required here as much as possible.
# Prefer environments that run with default arguments.
ENV_ARG_OVERRIDES: dict[str, dict] = {
    # *_wbc_pink embodiments expect a Pink IK action space and are incompatible
    # with the zero_action policy; switch each to the joint-control variant of
    # the same robot family for the smoke test.
    "galileo_g1_locomanip_pick_and_place": {"embodiment": "g1_wbc_joint"},
    "galileo_pick_and_place": {"embodiment": "gr1_joint"},
    "gr1_open_microwave": {"embodiment": "gr1_joint"},
    "gr1_put_and_close_door": {"embodiment": "gr1_joint"},
    "gr1_turn_stand_mixer_knob": {"embodiment": "gr1_joint"},
    "put_item_in_fridge_and_close_door": {"embodiment": "gr1_joint"},
}

# These environments require a custom physics configuration and have dedicated
# runtime coverage in the matching backend phase. Running them in this mixed
# subprocess would retain their backend allocations before later jobs start.
DEDICATED_RUNTIME_ENVIRONMENTS = {
    "yam_cable_routing": "test_yam_cable_routing.py (with_newton)",
}


def _build_jobs_for_all_envs() -> list[dict]:
    ensure_environments_registered()
    env_names = sorted(EnvironmentRegistry().get_all_keys())
    jobs = []
    for env_name in env_names:
        if env_name in DEDICATED_RUNTIME_ENVIRONMENTS:
            continue
        arena_env_args = {"environment": env_name}
        arena_env_args.update(ENV_ARG_OVERRIDES.get(env_name, {}))
        jobs.append({
            "name": f"smoke_{env_name}",
            "arena_env_args": arena_env_args,
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_config_dict": {},
        })
    return jobs


@pytest.mark.with_subprocess
def test_experiment_runner_all_environments(tmp_path):
    """Boot every generic-backend environment with the zero_action policy."""
    jobs = _build_jobs_for_all_envs()
    assert len(jobs) > 0, "Expected at least one environment to be registered"

    config_path = str(tmp_path / "test_experiment_runner_all_environments.json")
    write_jobs_config_to_file(jobs, config_path)
    run_experiment_runner(config_path, headless=HEADLESS)


def test_all_environments_have_default_args():
    """Every registered environment's generated CLI flags must have defaults.

    Enforces that environment configs declare defaults for all generated CLI
    options. Each failing environment is reported rather than aborting early.
    """
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()

    failures: list[str] = []
    for env_name in sorted(env_registry.get_all_keys()):
        environment_factory_type = env_registry.get_component_by_name(env_name)
        # Use a parser that doesn't sys.exit on error so we can collect failures.
        parser = argparse.ArgumentParser(exit_on_error=False)
        add_environment_cli_args(parser, environment_factory_type)

        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if action.required:
                failures.append(f"{env_name}: --{action.dest} is required (no default)")

        try:
            parser.parse_args([])
        except (argparse.ArgumentError, SystemExit) as e:
            failures.append(f"{env_name}: parse_args([]) failed: {e}")

    assert not failures, "Environments cannot be run with default args:\n" + "\n".join(failures)
