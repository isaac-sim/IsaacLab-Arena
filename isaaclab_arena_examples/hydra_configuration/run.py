# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose typed Hydra experiments and dispatch them through Arena's evaluation APIs.

Hydra composes the core ``ArenaExperimentCfg`` type. Explicit compatibility
adapters translate experiments to the ``Job`` and ``argparse.Namespace``
objects still consumed by the current evaluation loop.
"""

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
from isaaclab_arena.evaluation.arena_experiment_cfg import ArenaExperimentCfg
from isaaclab_arena.evaluation.eval_runner import evaluate_jobs
from isaaclab_arena.evaluation.job_manager import Job
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Hydra composition


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


def compose_hydra_example_experiments(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> list[ArenaExperimentCfg]:
    """Compose every YAML entry as an independent typed experiment.

    Args:
        config_path: Path to the YAML document containing the ordered experiment list.
        overrides: Optional Hydra override tokens applied independently to every experiment.

    Returns:
        The typed experiments in dispatch order.
    """
    path = Path(config_path).resolve()
    assert path.is_file(), f"Hydra example config does not exist: {path}"
    assert path.suffix == ".yaml", f"Hydra example config must be a .yaml file: {path}"
    config_name = path.stem.replace("-", "_")
    experiment_schema_config_name = f"_{config_name}_experiment_schema"
    hydra_config_name_prefix = f"_{config_name}_experiment"

    document = OmegaConf.load(path)
    assert isinstance(document, DictConfig), "Hydra example config must be a mapping"
    assert set(document) == {"experiments"}, "Hydra example config must contain only an 'experiments' list"
    experiment_nodes = document["experiments"]
    assert isinstance(experiment_nodes, ListConfig), "experiments must be a list"
    assert experiment_nodes, "experiments must not be empty"

    environment_providers = _typed_environment_providers()
    config_store = ConfigStore.instance()
    config_store.store(name=experiment_schema_config_name, node=ArenaExperimentCfg)
    for environment_name, provider in environment_providers.items():
        cfg_type = provider.cfg_type
        assert cfg_type is not None
        config_store.store(group="environment", name=environment_name, node=cfg_type)

    hydra_experiment_config_names = []
    for index, experiment_node in enumerate(experiment_nodes):
        assert isinstance(experiment_node, DictConfig), f"experiments[{index}] must be a mapping"
        environment = experiment_node.get("environment")
        assert isinstance(environment, DictConfig), f"experiments[{index}].environment must be a mapping"
        environment_name = environment.get("name")
        assert isinstance(environment_name, str), f"experiments[{index}].environment.name must be a string"
        assert (
            environment_name in environment_providers
        ), f"experiments[{index}].environment.name '{environment_name}' does not provide typed configuration"

        experiment_data = OmegaConf.to_container(experiment_node, resolve=False)
        assert isinstance(experiment_data, dict)
        hydra_experiment_node = {
            "defaults": [
                experiment_schema_config_name,
                {"environment": environment_name},
                "_self_",
            ],
            **experiment_data,
        }
        hydra_experiment_config_name = f"{hydra_config_name_prefix}_{index}"
        config_store.store(name=hydra_experiment_config_name, node=hydra_experiment_node)
        hydra_experiment_config_names.append(hydra_experiment_config_name)

    experiments = []
    with initialize(version_base=None, config_path=None):
        for experiment_config_name in hydra_experiment_config_names:
            experiment = OmegaConf.to_object(
                compose(config_name=experiment_config_name, overrides=list(overrides or []))
            )
            assert isinstance(experiment, ArenaExperimentCfg)
            provider = environment_providers.get(experiment.environment.name)
            assert (
                provider is not None
            ), f"Environment '{experiment.environment.name}' is not registered with a typed config"
            assert provider.cfg_type is not None
            assert isinstance(experiment.environment, provider.cfg_type)
            experiments.append(experiment)

    experiment_names = [experiment.name for experiment in experiments]
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"
    return experiments


# Existing evaluation API adapters
# Keep these conversions at the edge so Hydra composition remains typed.


def _arena_experiment_cfg_to_legacy_builder_namespace(
    experiment_configuration: ArenaExperimentCfg,
    device: str,
    language_instruction: str | None = None,
) -> argparse.Namespace:
    """Translate a typed experiment to the Namespace still required by ArenaEnvBuilder."""
    builder = experiment_configuration.environment_builder
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


def _arena_experiment_cfg_to_eval_job(experiment_configuration: ArenaExperimentCfg) -> Job:
    """Translate a typed experiment to eval_runner's existing ``Job`` model."""
    return Job(
        name=experiment_configuration.name,
        num_envs=experiment_configuration.environment_builder.num_envs,
        arena_env_args=[],
        policy_type=experiment_configuration.policy.type,
        policy_config_dict=experiment_configuration.policy.parameters,
        num_steps=experiment_configuration.rollout.num_steps,
        num_rebuilds=experiment_configuration.num_rebuilds,
        variations=Job.convert_variations_dict_to_hydra_overrides(experiment_configuration.variations),
    )


# Dispatcher integration
# Process-wide settings and runtime construction are dispatcher responsibilities.


def _resolve_legacy_eval_runner_namespace(
    experiment_configurations: list[ArenaExperimentCfg],
    device: str,
    visualizer: str | None,
) -> argparse.Namespace:
    """Resolve process-wide settings in the Namespace expected by eval_runner."""
    return argparse.Namespace(
        device=device,
        enable_cameras=any(experiment.environment.enable_cameras for experiment in experiment_configurations),
        visualizer=visualizer,
        num_steps=None,
        output_base_dir="/eval/output",
        record_viewport_video=False,
        record_camera_video=False,
        serve_evaluation_report=False,
        evaluation_report_port=8000,
        continue_on_error=False,
    )


def _load_environment_from_arena_experiment_cfg(
    experiment_configurations_by_name: dict[str, ArenaExperimentCfg],
    device: str,
    job: Job,
    render_mode: str | None,
):
    """Build one typed environment through the existing ArenaEnvBuilder API."""
    # Isaac Sim must be launched before importing modules that load USD/PhysX bindings.
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    experiment_configuration = experiment_configurations_by_name[job.name]
    environment_configuration = experiment_configuration.environment
    environment_providers = _typed_environment_providers()
    assert environment_configuration.name in environment_providers
    environment_provider = environment_providers[environment_configuration.name]()
    assert environment_provider.cfg_type is not None
    assert isinstance(environment_configuration, environment_provider.cfg_type)
    arena_environment = environment_provider.build(environment_configuration)
    legacy_builder_namespace = _arena_experiment_cfg_to_legacy_builder_namespace(
        experiment_configuration,
        device,
        job.language_instruction,
    )
    arena_builder = ArenaEnvBuilder(
        arena_environment,
        legacy_builder_namespace,
        hydra_overrides=job.variations,
    )
    _, env_cfg, env_kwargs = arena_builder.build_registered()
    if env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job.name}"
    return arena_builder.make_registered(env_cfg, env_kwargs, render_mode=render_mode)


def run(
    experiment_configurations: list[ArenaExperimentCfg],
    *,
    device: str = "cuda:0",
    visualizer: str | None = None,
) -> None:
    """Dispatch the configured experiments sequentially through one simulation app."""
    experiment_names = [experiment.name for experiment in experiment_configurations]
    assert experiment_names, "experiments must not be empty"
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"
    print(f"[hydra-example] experiments={experiment_names}", flush=True)

    eval_runner_namespace = _resolve_legacy_eval_runner_namespace(
        experiment_configurations,
        device,
        visualizer,
    )
    eval_jobs = [_arena_experiment_cfg_to_eval_job(experiment) for experiment in experiment_configurations]
    experiment_configurations_by_name = {experiment.name: experiment for experiment in experiment_configurations}
    with SimulationAppContext(eval_runner_namespace):
        environment_loader = functools.partial(
            _load_environment_from_arena_experiment_cfg,
            experiment_configurations_by_name,
            eval_runner_namespace.device,
        )
        evaluate_jobs(
            eval_runner_namespace,
            eval_jobs,
            environment_loader=environment_loader,
        )
        completed_job_names = [job.name for job in eval_jobs]
        print(f"[hydra-example] completed jobs={completed_job_names}", flush=True)


def compose_from_command_line(
    arguments: list[str],
) -> tuple[list[ArenaExperimentCfg], argparse.Namespace]:
    """Parse dispatcher flags and compose every experiment from the remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path, help="YAML file containing the experiments to dispatch.")
    parser.add_argument("--device", default="cuda:0", help="Isaac Sim device used by this dispatcher.")
    parser.add_argument("--visualizer", "--viz", help="Isaac Lab visualizer backend, for example 'kit'.")
    # argparse owns only this frontend's process flags. All remaining tokens are Hydra overrides.
    dispatcher_arguments, hydra_overrides = parser.parse_known_args(arguments)
    experiments = compose_hydra_example_experiments(dispatcher_arguments.config_path, hydra_overrides)
    return experiments, dispatcher_arguments


def main() -> None:
    """Compose the portable experiments and dispatch them sequentially."""
    experiment_configurations, dispatcher_arguments = compose_from_command_line(sys.argv[1:])
    run(
        experiment_configurations,
        device=dispatcher_arguments.device,
        visualizer=dispatcher_arguments.visualizer,
    )


if __name__ == "__main__":
    main()
