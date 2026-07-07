# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Translate the existing eval-jobs JSON format into typed experiment configs.

``adapt_legacy_eval_config`` is the boundary between the old JSON/argparse input
and typed experiment execution. Everything else in this module is compatibility
code hidden from the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, RolloutCfg
from isaaclab_arena.evaluation.job_manager import Job
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_environments.cli import (
    ensure_environments_registered,
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

# TODO(cvolk, 2026-07-07): Delete this adapter when eval_runner loads typed YAML
# experiments directly. The current JSON format identifies environments and policies
# by name and mixes environment-specific and builder values in ``arena_env_args``.
# This module resolves those names, separates the values into their concrete typed
# configs, and keeps graph or CLI-only environments on the existing argparse path.
# The planned Hydra/YAML frontend will compose those typed configs directly, so none
# of this JSON translation or argparse fallback will be needed.


@dataclass
class _LegacyCliEnvironmentCfg(ArenaEnvironmentCfg):
    """Carry environment arguments understood only by the existing CLI adapter."""

    arguments: list[str]
    """Arguments passed to the existing environment parser."""


@dataclass(frozen=True)
class _EnvironmentAdaptation:
    """Hold the environment parts extracted from legacy ``arena_env_args``."""

    cfg: ArenaEnvironmentCfg
    """Concrete environment config, or a private argparse compatibility config."""

    builder_cfg: ArenaEnvBuilderCfg
    """Arena builder values that were mixed into the legacy environment arguments."""


def adapt_legacy_eval_config(
    config: dict[str, Any],
    *,
    device: str,
) -> list[ArenaExperimentCfg]:
    """Translate a legacy eval-jobs document into typed experiment configs."""
    assert set(config) == {"jobs"}, "legacy evaluation config must contain only a 'jobs' list"
    job_configs = config["jobs"]
    assert isinstance(job_configs, list), "legacy evaluation config 'jobs' must be a list"

    experiment_cfgs = [_adapt_legacy_job(job_config, device=device) for job_config in job_configs]
    experiment_names = [experiment_cfg.name for experiment_cfg in experiment_cfgs]
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"
    return experiment_cfgs


def _adapt_legacy_job(
    job_config: dict[str, Any],
    *,
    device: str,
) -> ArenaExperimentCfg:
    """Translate one legacy job mapping into a typed experiment."""
    assert isinstance(job_config, dict), "each legacy job must be a mapping"
    for required_field in ("name", "arena_env_args", "policy_type"):
        assert required_field in job_config, f"{required_field} is required"

    adapted_environment = _adapt_environment(
        job_config["arena_env_args"],
        device=device,
        language_instruction=job_config.get("language_instruction"),
    )
    variations = job_config.get("variations", {})
    assert isinstance(variations, dict), "variations must be a mapping"

    return ArenaExperimentCfg(
        name=job_config["name"],
        environment=adapted_environment.cfg,
        environment_builder=adapted_environment.builder_cfg,
        policy=_adapt_policy_cfg(job_config),
        rollout=RolloutCfg(
            num_steps=job_config.get("num_steps"),
            num_episodes=job_config.get("num_episodes"),
        ),
        num_rebuilds=job_config.get("num_rebuilds", 1),
        variations=variations,
    )


def _adapt_environment(
    arena_env_args: dict[str, Any],
    *,
    device: str,
    language_instruction: str | None,
) -> _EnvironmentAdaptation:
    """Split legacy environment arguments into typed configuration and construction."""
    assert isinstance(arena_env_args, dict), "arena_env_args must be a mapping"
    assert arena_env_args.get("environment"), "arena_env_args.environment is required"

    environment_source = str(arena_env_args["environment"])
    builder_cfg = _adapt_builder_cfg(arena_env_args, device, language_instruction)

    if environment_source.endswith((".yaml", ".yml")):
        return _preserve_argparse_environment(arena_env_args, builder_cfg)

    return _adapt_registered_environment(arena_env_args, environment_source, builder_cfg)


def _adapt_builder_cfg(
    arena_env_args: dict[str, Any],
    device: str,
    language_instruction: str | None,
) -> ArenaEnvBuilderCfg:
    """Extract Arena builder values that the legacy format mixes with environment values."""
    builder_values = {
        config_field.name: arena_env_args[config_field.name]
        for config_field in fields(ArenaEnvBuilderCfg)
        if config_field.name in arena_env_args
    }
    builder_values["device"] = device
    builder_values["language_instruction"] = language_instruction
    return ArenaEnvBuilderCfg(**builder_values)


def _adapt_registered_environment(
    arena_env_args: dict[str, Any],
    environment_name: str,
    builder_cfg: ArenaEnvBuilderCfg,
) -> _EnvironmentAdaptation:
    """Use typed configs for a registered environment when all legacy fields are known."""
    ensure_environments_registered()
    environment_registry = EnvironmentRegistry()
    assert environment_registry.is_registered(
        environment_name, ensure_loaded=False
    ), f"Environment {environment_name!r} is not registered"
    environment_factory_type = environment_registry.get_component_by_name(environment_name)
    environment_cfg_type = environment_registry.get_environment_cfg_type(environment_factory_type)
    builder_field_names = {config_field.name for config_field in fields(ArenaEnvBuilderCfg)}
    environment_field_names = {config_field.name for config_field in fields(environment_cfg_type)}
    supported_argument_names = builder_field_names | environment_field_names | {"environment"}

    # Preserve any environment-specific CLI option not represented by the typed
    # configuration until the JSON/argparse frontend is removed.
    if set(arena_env_args) - supported_argument_names:
        return _preserve_argparse_environment(arena_env_args, builder_cfg)

    environment_values = {
        field_name: value for field_name, value in arena_env_args.items() if field_name in environment_field_names
    }
    environment_cfg = environment_cfg_type(**environment_values)
    return _EnvironmentAdaptation(
        cfg=environment_cfg,
        builder_cfg=builder_cfg,
    )


def _preserve_argparse_environment(
    arena_env_args: dict[str, Any],
    builder_cfg: ArenaEnvBuilderCfg,
) -> _EnvironmentAdaptation:
    """Keep unsupported environment fields inside the existing CLI path."""
    environment_cfg = _LegacyCliEnvironmentCfg(
        arguments=Job.convert_args_dict_to_cli_args_list(arena_env_args),
    )
    return _EnvironmentAdaptation(
        cfg=environment_cfg,
        builder_cfg=builder_cfg,
    )


def _build_arena_builder_from_legacy_cfg(experiment: ArenaExperimentCfg) -> ArenaEnvBuilder:
    """Build an Arena environment builder from preserved legacy CLI arguments."""
    assert isinstance(experiment.environment, _LegacyCliEnvironmentCfg)
    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(experiment.environment.arguments)
    args_cli.device = experiment.environment_builder.device
    args_cli.language_instruction = experiment.environment_builder.language_instruction
    return get_arena_builder_from_cli(
        args_cli,
        hydra_overrides=Job.convert_variations_dict_to_hydra_overrides(experiment.variations),
    )


def _adapt_policy_cfg(job_config: dict[str, Any]) -> PolicyCfg:
    """Construct a concrete policy config from legacy policy fields."""
    assert not (
        "policy_config_dict" in job_config and "policy_args" in job_config
    ), "provide only policy_config_dict; policy_args is a deprecated alias"
    policy_values = job_config.get("policy_config_dict", job_config.get("policy_args", {}))
    assert isinstance(policy_values, dict), "policy_config_dict must be a mapping"

    policy_type = get_policy_cls(job_config["policy_type"])
    policy_cfg_type = PolicyRegistry().get_policy_cfg_type(policy_type)
    return policy_cfg_type(**policy_values)
