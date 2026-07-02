# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose ``hydra_example_suite.yaml`` and evaluate its single job."""

from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import asdict

from isaaclab_arena.evaluation.eval_runner import evaluate_jobs
from isaaclab_arena.evaluation.job_manager import Job
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_examples.hydra_configuration.config import ArenaRunConfiguration, compose_hydra_example_suite


def _environment_builder_arguments(
    configuration: ArenaRunConfiguration,
    language_instruction: str | None = None,
    device: str | None = None,
) -> argparse.Namespace:
    """Adapt the typed example configuration to the current Arena builder API."""
    builder = configuration.environment_builder
    return argparse.Namespace(
        num_envs=builder.num_envs,
        env_spacing=builder.env_spacing,
        seed=builder.seed,
        solve_relations=builder.solve_relations,
        placement_seed=builder.placement_seed,
        resolve_on_reset=builder.resolve_on_reset,
        random_yaw_init=builder.random_yaw_init,
        disable_fabric=builder.disable_fabric,
        mimic=builder.mimic,
        presets=builder.presets,
        device=device or configuration.simulation_app.device,
        language_instruction=language_instruction,
    )


def _evaluation_runner_arguments(configuration: ArenaRunConfiguration) -> argparse.Namespace:
    """Supply eval-runner defaults outside this MVP's YAML surface."""
    return argparse.Namespace(
        **asdict(configuration.simulation_app),
        num_steps=configuration.rollout.num_steps,
        output_base_dir="/eval/output",
        record_viewport_video=False,
        record_camera_video=False,
        serve_evaluation_report=False,
        evaluation_report_port=8000,
        continue_on_error=False,
    )


def _job_from_configuration(configuration: ArenaRunConfiguration) -> Job:
    """Create one eval job without translating the environment config through CLI tokens."""
    return Job(
        name=configuration.name,
        num_envs=configuration.environment_builder.num_envs,
        arena_env_args=[],
        policy_type=configuration.policy.type,
        policy_config_dict=configuration.policy.parameters,
        num_steps=configuration.rollout.num_steps,
    )


def _load_environment_from_configuration(
    configuration: ArenaRunConfiguration,
    enable_cameras: bool,
    device: str,
    job: Job,
    render_mode: str | None,
):
    """Build one typed environment for the existing eval-runner loop."""
    # Isaac Sim must be launched before importing modules that load USD/PhysX bindings.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_environment = configuration.environment.build(enable_cameras=enable_cameras)
    arena_builder = ArenaEnvBuilder(
        arena_environment,
        _environment_builder_arguments(configuration, job.language_instruction, device),
        hydra_overrides=job.variations,
    )
    _, env_cfg, env_kwargs = arena_builder.build_registered()
    if env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job.name}"
    return arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)


def run(configuration: ArenaRunConfiguration) -> None:
    """Evaluate the composed Hydra run with Arena's existing evaluation engine."""
    print(
        f"[hydra-example] environment={type(configuration.environment).__name__} "
        f"steps={configuration.rollout.num_steps}",
        flush=True,
    )
    args_cli = _evaluation_runner_arguments(configuration)
    job = _job_from_configuration(configuration)
    with SimulationAppContext(args_cli) as simulation_app:
        # AppLauncher may resolve process settings from environment variables (for example,
        # ENABLE_CAMERAS), so environment construction must use the resolved launch values.
        resolved_enable_cameras = simulation_app.app_launcher._enable_cameras  # noqa: SLF001
        environment_loader = functools.partial(
            _load_environment_from_configuration,
            configuration,
            resolved_enable_cameras,
            args_cli.device,
        )
        evaluate_jobs(args_cli, [job], environment_loader=environment_loader)
        print(f"[hydra-example] completed '{configuration.name}'", flush=True)


def compose_from_command_line(arguments: list[str]) -> ArenaRunConfiguration:
    """Compose the run config from ``--viz`` plus Hydra override tokens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualizer", "--viz", help="Isaac Lab visualizer backend, for example 'kit'.")
    launcher_arguments, hydra_overrides = parser.parse_known_args(arguments)
    if launcher_arguments.visualizer is not None:
        hydra_overrides.append(f"simulation_app.visualizer={launcher_arguments.visualizer}")
    return compose_hydra_example_suite(hydra_overrides)


def main() -> None:
    """Compose and run the co-located suite."""
    run(compose_from_command_line(sys.argv[1:]))


if __name__ == "__main__":
    main()
