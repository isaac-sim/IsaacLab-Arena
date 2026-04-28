# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test every registered environment via eval_runner.

Auto-discovers environments from the EnvironmentRegistry and runs each one for
a few steps with the zero_action policy in a single eval_runner subprocess.
The test passes if no environment errors out during startup or stepping.
"""

import argparse

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.tests.test_eval_runner import run_eval_runner, write_jobs_config_to_file
from isaaclab_arena_environments.cli import ensure_environments_registered

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
}


def _build_jobs_for_all_envs() -> list[dict]:
    ensure_environments_registered()
    env_names = sorted(EnvironmentRegistry().get_all_keys())
    jobs = []
    for env_name in env_names:
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
def test_eval_runner_all_environments(tmp_path):
    """Boot every registered environment for a few steps with the zero_action policy."""
    jobs = _build_jobs_for_all_envs()
    assert len(jobs) > 0, "Expected at least one environment to be registered"

    config_path = str(tmp_path / "test_eval_runner_all_environments.json")
    write_jobs_config_to_file(jobs, config_path)
    run_eval_runner(config_path, headless=HEADLESS)


def test_all_environments_have_default_args():
    """Every registered env's ``add_cli_args`` parser must accept zero extra args.

    Enforces that envs declare defaults for all their CLI options so that they
    can be run without explicit arguments. Each failing env is reported in the
    assertion message rather than aborting on the first one.
    """
    ensure_environments_registered()
    env_registry = EnvironmentRegistry()

    failures: list[str] = []
    for env_name in sorted(env_registry.get_all_keys()):
        env_cls = env_registry.get_component_by_name(env_name)
        # Use a parser that doesn't sys.exit on error so we can collect failures.
        parser = argparse.ArgumentParser(exit_on_error=False)
        env_cls.add_cli_args(parser)

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
