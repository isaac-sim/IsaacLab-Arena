# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert the existing eval-jobs JSON format into typed experiment configs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, ArenaExperimentCollectionCfg, RolloutCfg
from isaaclab_arena.evaluation.legacy_environment_cli_args import legacy_environment_args_to_cli_args
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_environments.cli import ensure_environments_registered

# TODO(cvolk, 2026-07-07): Delete this adapter after the remaining eval-jobs JSON
# documents move to typed YAML collections. The JSON format identifies environments
# and policies by name and mixes environment-specific and builder values in
# ``arena_env_args``.
# This module resolves those names, separates the values into their concrete typed
# configs, and marks graph-YAML environments for the narrow argparse compatibility
# path in ``legacy_graph_environment_cli``.
# Typed YAML collections compose those configs directly; this translation remains only
# for the JSON configurations that have not migrated yet.


@dataclass(frozen=True)
class _EnvironmentCfgs:
    """Hold the two typed configs extracted from legacy ``arena_env_args``."""

    environment_cfg: ArenaEnvironmentCfg
    """Concrete environment config, or the temporary graph-YAML compatibility config."""

    environment_builder_cfg: ArenaEnvBuilderCfg
    """Arena builder values that were mixed into the legacy environment arguments."""


def experiment_collection_from_legacy_eval_config(
    config: dict[str, Any],
    device: str,
) -> ArenaExperimentCollectionCfg:
    """Create a typed experiment collection from a legacy eval-jobs document."""
    assert set(config) == {"jobs"}, "legacy evaluation config must contain only a 'jobs' list"
    job_configs = config["jobs"]
    assert isinstance(job_configs, list), "legacy evaluation config 'jobs' must be a list"

    experiment_cfgs = [_experiment_cfg_from_legacy_job(job_config, device=device) for job_config in job_configs]
    experiment_names = [experiment_cfg.name for experiment_cfg in experiment_cfgs]
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"
    return ArenaExperimentCollectionCfg(
        experiments=dict(zip(experiment_names, experiment_cfgs, strict=True)),
    )


def _experiment_cfg_from_legacy_job(
    job_config: dict[str, Any],
    device: str,
) -> ArenaExperimentCfg:
    """Create one typed experiment config from a legacy job mapping."""
    assert isinstance(job_config, dict), "each legacy job must be a mapping"
    for required_field in ("name", "arena_env_args", "policy_type"):
        assert required_field in job_config, f"{required_field} is required"

    environment_cfgs = _environment_cfgs_from_legacy_args(
        job_config["arena_env_args"],
        device=device,
        language_instruction=job_config.get("language_instruction"),
    )
    variations = job_config.get("variations", {})
    assert isinstance(variations, dict), "variations must be a mapping"

    return ArenaExperimentCfg(
        name=job_config["name"],
        environment=environment_cfgs.environment_cfg,
        environment_builder=environment_cfgs.environment_builder_cfg,
        policy=_policy_cfg_from_legacy_job(job_config),
        rollout=RolloutCfg(
            num_steps=job_config.get("num_steps"),
            num_episodes=job_config.get("num_episodes"),
        ),
        num_rebuilds=job_config.get("num_rebuilds", 1),
        variations=variations,
    )


def _environment_cfgs_from_legacy_args(
    arena_env_args: dict[str, Any],
    device: str,
    language_instruction: str | None,
) -> _EnvironmentCfgs:
    """Split legacy environment arguments into environment and builder configs."""
    assert isinstance(arena_env_args, dict), "arena_env_args must be a mapping"
    assert arena_env_args.get("environment"), "arena_env_args.environment is required"

    environment_source = str(arena_env_args["environment"])
    environment_cfg = (
        _graph_environment_cfg_from_legacy_args(arena_env_args)
        if environment_source.endswith((".yaml", ".yml"))
        else _registered_environment_cfg_from_legacy_args(arena_env_args, environment_source)
    )
    return _EnvironmentCfgs(
        environment_cfg=environment_cfg,
        environment_builder_cfg=_builder_cfg_from_legacy_args(arena_env_args, device, language_instruction),
    )


def _builder_cfg_from_legacy_args(
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


def _registered_environment_cfg_from_legacy_args(
    arena_env_args: dict[str, Any],
    environment_name: str,
) -> ArenaEnvironmentCfg:
    """Create a registered environment config from legacy arguments."""
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

    unknown_argument_names = set(arena_env_args) - supported_argument_names
    assert (
        not unknown_argument_names
    ), f"Environment {environment_name!r} has no typed configuration fields for {sorted(unknown_argument_names)}"

    environment_values = {
        field_name: value for field_name, value in arena_env_args.items() if field_name in environment_field_names
    }
    return environment_cfg_type(**environment_values)


def _graph_environment_cfg_from_legacy_args(
    arena_env_args: dict[str, Any],
) -> LegacyGraphEnvironmentCfg:
    """Create the temporary graph-YAML compatibility config from legacy arguments."""
    return LegacyGraphEnvironmentCfg(
        arena_env_args=legacy_environment_args_to_cli_args(arena_env_args),
    )


def _policy_cfg_from_legacy_job(job_config: dict[str, Any]) -> PolicyCfg:
    """Construct a concrete policy config from legacy policy fields."""
    assert not (
        "policy_config_dict" in job_config and "policy_args" in job_config
    ), "provide only policy_config_dict; policy_args is a deprecated alias"
    policy_values = job_config.get("policy_config_dict", job_config.get("policy_args", {}))
    assert isinstance(policy_values, dict), "policy_config_dict must be a mapping"

    policy_type = get_policy_cls(job_config["policy_type"])
    policy_cfg_type = PolicyRegistry().get_policy_cfg_type(policy_type)
    return policy_cfg_type(**policy_values)
