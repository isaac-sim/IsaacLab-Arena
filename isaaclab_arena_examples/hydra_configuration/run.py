# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose portable jobs from a required YAML configuration and dispatch them sequentially."""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_job_cfg import ArenaJobCfg
from isaaclab_arena.evaluation.eval_runner import evaluate_jobs
from isaaclab_arena.evaluation.job_manager import Job
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


def _typed_environment_providers() -> dict[str, type[ExampleEnvironmentBase]]:
    """Return registered environment providers that expose a structured config type."""
    ensure_environments_registered()
    registry = EnvironmentRegistry()
    providers = {}
    for environment_name in registry.get_all_keys():
        provider = registry.get_component_by_name(environment_name)
        cfg_type = getattr(provider, "cfg_type", None)
        if cfg_type is None:
            continue
        assert issubclass(cfg_type, ArenaEnvironmentCfg)
        providers[environment_name] = provider
    return providers


def compose_hydra_example_jobs(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> list[ArenaJobCfg]:
    """Compose every YAML entry as an independent typed job.

    Args:
        config_path: Path to the YAML document containing the ordered job list.
        overrides: Optional Hydra override tokens applied independently to every job.

    Returns:
        The typed jobs in dispatch order.
    """
    path = Path(config_path).resolve()
    assert path.is_file(), f"Hydra example config does not exist: {path}"
    assert path.suffix == ".yaml", f"Hydra example config must be a .yaml file: {path}"
    config_name = path.stem.replace("-", "_")
    job_schema_config_name = f"_{config_name}_job_schema"
    runtime_config_name = f"_{config_name}_job"

    document = OmegaConf.load(path)
    assert isinstance(document, DictConfig), "Hydra example config must be a mapping"
    assert set(document) == {"jobs"}, "Hydra example config must contain only a 'jobs' list"
    job_nodes = document["jobs"]
    assert isinstance(job_nodes, ListConfig), "jobs must be a list"
    assert job_nodes, "jobs must not be empty"

    environment_providers = _typed_environment_providers()
    config_store = ConfigStore.instance()
    config_store.store(name=job_schema_config_name, node=ArenaJobCfg)
    for environment_name, provider in environment_providers.items():
        cfg_type = provider.cfg_type
        assert cfg_type is not None
        config_store.store(group="environment", name=environment_name, node=cfg_type)

    config_names = []
    for index, job_node in enumerate(job_nodes):
        assert isinstance(job_node, DictConfig), f"jobs[{index}] must be a mapping"
        environment = job_node.get("environment")
        assert isinstance(environment, DictConfig), f"jobs[{index}].environment must be a mapping"
        environment_name = environment.get("name")
        assert isinstance(environment_name, str), f"jobs[{index}].environment.name must be a string"
        assert (
            environment_name in environment_providers
        ), f"jobs[{index}].environment.name '{environment_name}' does not provide typed configuration"

        job_data = OmegaConf.to_container(job_node, resolve=False)
        assert isinstance(job_data, dict)
        runtime_job = {
            "defaults": [
                job_schema_config_name,
                {"environment": environment_name},
                "_self_",
            ],
            **job_data,
        }
        runtime_job_config_name = f"{runtime_config_name}_{index}"
        config_store.store(name=runtime_job_config_name, node=runtime_job)
        config_names.append(runtime_job_config_name)

    jobs = []
    with initialize(version_base=None, config_path=None):
        for config_name in config_names:
            job = OmegaConf.to_object(compose(config_name=config_name, overrides=list(overrides or [])))
            assert isinstance(job, ArenaJobCfg)
            provider = environment_providers.get(job.environment.name)
            assert provider is not None, f"Environment '{job.environment.name}' is not registered with a typed config"
            assert provider.cfg_type is not None
            assert isinstance(job.environment, provider.cfg_type)
            jobs.append(job)

    job_names = [job.name for job in jobs]
    assert len(job_names) == len(set(job_names)), "job names must be unique"
    return jobs


def _environment_builder_arguments(
    job_configuration: ArenaJobCfg,
    device: str,
    language_instruction: str | None = None,
) -> argparse.Namespace:
    """Adapt the typed example configuration to the current Arena builder API."""
    builder = job_configuration.environment_builder
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
        device=device,
        language_instruction=language_instruction,
    )


def _evaluation_runner_arguments(
    job_configurations: list[ArenaJobCfg],
    device: str,
    visualizer: str | None,
) -> argparse.Namespace:
    """Resolve this dispatcher's process-wide settings from its jobs and CLI."""
    return argparse.Namespace(
        device=device,
        enable_cameras=any(job.environment.enable_cameras for job in job_configurations),
        visualizer=visualizer,
        num_steps=None,
        output_base_dir="/eval/output",
        record_viewport_video=False,
        record_camera_video=False,
        serve_evaluation_report=False,
        evaluation_report_port=8000,
        continue_on_error=False,
    )


def _job_from_configuration(job_configuration: ArenaJobCfg) -> Job:
    """Create one eval job without translating the environment config through CLI tokens."""
    return Job(
        name=job_configuration.name,
        num_envs=job_configuration.environment_builder.num_envs,
        arena_env_args=[],
        policy_type=job_configuration.policy.type,
        policy_config_dict=job_configuration.policy.parameters,
        num_steps=job_configuration.rollout.num_steps,
        num_rebuilds=job_configuration.num_rebuilds,
        variations=Job.convert_variations_dict_to_hydra_overrides(job_configuration.variations),
    )


def _load_environment_from_configuration(
    job_configurations_by_name: dict[str, ArenaJobCfg],
    device: str,
    job: Job,
    render_mode: str | None,
):
    """Build one typed environment for the existing eval-runner loop."""
    # Isaac Sim must be launched before importing modules that load USD/PhysX bindings.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    job_configuration = job_configurations_by_name[job.name]
    environment_configuration = job_configuration.environment
    environment_providers = _typed_environment_providers()
    assert environment_configuration.name in environment_providers
    environment_provider = environment_providers[environment_configuration.name]()
    assert environment_provider.cfg_type is not None
    assert isinstance(environment_configuration, environment_provider.cfg_type)
    arena_environment = environment_provider.build(environment_configuration)
    arena_builder = ArenaEnvBuilder(
        arena_environment,
        _environment_builder_arguments(job_configuration, device, job.language_instruction),
        hydra_overrides=job.variations,
    )
    _, env_cfg, env_kwargs = arena_builder.build_registered()
    if env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job.name}"
    return arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)


def run(
    job_configurations: list[ArenaJobCfg],
    *,
    device: str = "cuda:0",
    visualizer: str | None = None,
) -> None:
    """Dispatch the configured jobs sequentially through one simulation app."""
    job_names = [job.name for job in job_configurations]
    assert job_names, "jobs must not be empty"
    assert len(job_names) == len(set(job_names)), "job names must be unique"
    print(f"[hydra-example] jobs={job_names}", flush=True)

    args_cli = _evaluation_runner_arguments(job_configurations, device, visualizer)
    jobs = [_job_from_configuration(job) for job in job_configurations]
    job_configurations_by_name = {job.name: job for job in job_configurations}
    with SimulationAppContext(args_cli):
        environment_loader = functools.partial(
            _load_environment_from_configuration,
            job_configurations_by_name,
            args_cli.device,
        )
        evaluate_jobs(args_cli, jobs, environment_loader=environment_loader)
        print(f"[hydra-example] completed jobs={job_names}", flush=True)


def compose_from_command_line(
    arguments: list[str],
) -> tuple[list[ArenaJobCfg], argparse.Namespace]:
    """Parse dispatcher flags and compose every job from the remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="YAML file containing the jobs to dispatch.")
    parser.add_argument("--device", default="cuda:0", help="Isaac Sim device used by this dispatcher.")
    parser.add_argument("--visualizer", "--viz", help="Isaac Lab visualizer backend, for example 'kit'.")
    launcher_arguments, hydra_overrides = parser.parse_known_args(arguments)
    return compose_hydra_example_jobs(launcher_arguments.config_path, hydra_overrides), launcher_arguments


def main() -> None:
    """Compose the portable jobs and dispatch them sequentially."""
    job_configurations, launcher_arguments = compose_from_command_line(sys.argv[1:])
    run(
        job_configurations,
        device=launcher_arguments.device,
        visualizer=launcher_arguments.visualizer,
    )


if __name__ == "__main__":
    main()
