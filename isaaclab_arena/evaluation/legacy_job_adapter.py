# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the existing eval-jobs JSON format to typed experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, RolloutCfg
from isaaclab_arena.evaluation.experiment_execution import ArenaBuilderFactory
from isaaclab_arena.evaluation.job_manager import Job
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_environments.cli import (
    _get_legacy_argparse_cfg_type,
    ensure_environments_registered,
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

# TODO(cvolk, 2026-07-06): Delete this module when eval_runner receives typed
# experiment configurations from the planned YAML frontend instead of JSON jobs.


@dataclass
class LegacyCliEnvironmentCfg(ArenaEnvironmentCfg):
    """Carry environment arguments understood only by the existing CLI adapter."""

    arguments: list[str]
    """Arguments passed to the existing environment parser."""


_BUILDER_FIELD_NAMES = {config_field.name for config_field in fields(ArenaEnvBuilderCfg)}


def load_legacy_experiments(
    config: dict[str, Any],
    *,
    device: str,
) -> tuple[list[ArenaExperimentCfg], dict[str, ArenaBuilderFactory]]:
    """Translate legacy jobs and resolve the builder used by each experiment."""
    assert set(config) == {"jobs"}, "legacy evaluation config must contain only a 'jobs' list"
    job_configs = config["jobs"]
    assert isinstance(job_configs, list), "legacy evaluation config 'jobs' must be a list"

    loaded_experiments = [_load_legacy_job(job_config, device=device) for job_config in job_configs]
    experiments = [experiment for experiment, _ in loaded_experiments]
    experiment_names = [experiment.name for experiment in experiments]
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"
    arena_builder_factories = {
        experiment.name: arena_builder_factory for experiment, arena_builder_factory in loaded_experiments
    }
    return experiments, arena_builder_factories


def _load_legacy_job(
    job_config: dict[str, Any],
    *,
    device: str,
) -> tuple[ArenaExperimentCfg, ArenaBuilderFactory]:
    """Translate one legacy job mapping into a typed experiment."""
    assert isinstance(job_config, dict), "each legacy job must be a mapping"
    for required_field in ("name", "arena_env_args", "policy_type"):
        assert required_field in job_config, f"{required_field} is required"

    environment_cfg, builder_cfg, arena_builder_factory = _load_environment(
        job_config["arena_env_args"],
        device=device,
        language_instruction=job_config.get("language_instruction"),
    )
    variations = job_config.get("variations", {})
    assert isinstance(variations, dict), "variations must be a mapping"

    experiment = ArenaExperimentCfg(
        name=job_config["name"],
        environment=environment_cfg,
        environment_builder=builder_cfg,
        policy=_load_policy_cfg(job_config),
        rollout=RolloutCfg(
            num_steps=job_config.get("num_steps"),
            num_episodes=job_config.get("num_episodes"),
        ),
        num_rebuilds=job_config.get("num_rebuilds", 1),
        variations=variations,
    )
    return experiment, arena_builder_factory


def _load_environment(
    arena_env_args: dict[str, Any],
    *,
    device: str,
    language_instruction: str | None,
) -> tuple[ArenaEnvironmentCfg, ArenaEnvBuilderCfg, ArenaBuilderFactory]:
    """Split legacy environment arguments into typed configuration and construction."""
    assert isinstance(arena_env_args, dict), "arena_env_args must be a mapping"
    assert arena_env_args.get("environment"), "arena_env_args.environment is required"

    environment_source = str(arena_env_args["environment"])
    builder_values = {
        field_name: value for field_name, value in arena_env_args.items() if field_name in _BUILDER_FIELD_NAMES
    }
    builder_values["device"] = device
    builder_values["language_instruction"] = language_instruction
    builder_cfg = ArenaEnvBuilderCfg(**builder_values)

    if environment_source.endswith((".yaml", ".yml")):
        return _legacy_cli_environment(arena_env_args, builder_cfg)

    ensure_environments_registered()
    environment_registry = EnvironmentRegistry()
    assert environment_registry.is_registered(
        environment_source, ensure_loaded=False
    ), f"Environment {environment_source!r} is not registered"
    environment_factory_type = environment_registry.get_component_by_name(environment_source)
    environment_cfg_type = _get_legacy_argparse_cfg_type(environment_factory_type)
    environment_field_names = {config_field.name for config_field in fields(environment_cfg_type)}
    typed_field_names = _BUILDER_FIELD_NAMES | environment_field_names | {"environment"}

    # Preserve any environment-specific CLI option not represented by the typed
    # configuration until the JSON/argparse frontend is removed.
    if set(arena_env_args) - typed_field_names:
        return _legacy_cli_environment(arena_env_args, builder_cfg)

    environment_values = {
        field_name: value for field_name, value in arena_env_args.items() if field_name in environment_field_names
    }
    environment_cfg = environment_cfg_type(**environment_values)
    arena_builder_factory = partial(
        _build_registered_arena_builder,
        environment_factory_type=environment_factory_type,
    )
    return environment_cfg, builder_cfg, arena_builder_factory


def _legacy_cli_environment(
    arena_env_args: dict[str, Any],
    builder_cfg: ArenaEnvBuilderCfg,
) -> tuple[ArenaEnvironmentCfg, ArenaEnvBuilderCfg, ArenaBuilderFactory]:
    """Keep unsupported environment fields inside the existing CLI path."""
    environment_cfg = LegacyCliEnvironmentCfg(
        arguments=Job.convert_args_dict_to_cli_args_list(arena_env_args),
    )
    return environment_cfg, builder_cfg, _build_legacy_cli_arena_builder


def _build_registered_arena_builder(
    experiment: ArenaExperimentCfg,
    *,
    environment_factory_type: type[ArenaEnvironmentFactory],
) -> ArenaEnvBuilder:
    """Build a registered environment using its already resolved factory."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_environment = environment_factory_type().build(experiment.environment)
    return ArenaEnvBuilder(
        arena_environment,
        experiment.environment_builder,
        hydra_overrides=Job.convert_variations_dict_to_hydra_overrides(experiment.variations),
    )


def _build_legacy_cli_arena_builder(experiment: ArenaExperimentCfg) -> ArenaEnvBuilder:
    """Build an environment through the existing argparse construction path."""
    assert isinstance(experiment.environment, LegacyCliEnvironmentCfg)
    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(experiment.environment.arguments)
    args_cli.device = experiment.environment_builder.device
    args_cli.language_instruction = experiment.environment_builder.language_instruction
    return get_arena_builder_from_cli(
        args_cli,
        hydra_overrides=Job.convert_variations_dict_to_hydra_overrides(experiment.variations),
    )


def _load_policy_cfg(job_config: dict[str, Any]) -> PolicyCfg:
    """Construct a concrete policy config from legacy policy fields."""
    assert not (
        "policy_config_dict" in job_config and "policy_args" in job_config
    ), "provide only policy_config_dict; policy_args is a deprecated alias"
    policy_values = job_config.get("policy_config_dict", job_config.get("policy_args", {}))
    assert isinstance(policy_values, dict), "policy_config_dict must be a mapping"

    policy_type = get_policy_cls(job_config["policy_type"])
    policy_cfg_type = PolicyRegistry().get_policy_cfg_type(policy_type)
    return policy_cfg_type(**policy_values)
