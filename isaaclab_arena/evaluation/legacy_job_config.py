# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Translate the existing eval-jobs dictionary format into typed experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, RolloutCfg
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena.utils.hydra_overrides import hydra_overrides_from_nested_dict
from isaaclab_arena_environments.cli import (
    ensure_environments_registered,
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

# TODO(cvolk, 2026-07-06): Delete this module after the evaluation configs, docs,
# tests, and chunk dispatcher migrate from the current JSON job schema to the
# structured YAML experiment schema.


# TODO(cvolk, 2026-07-06): Delete this opaque CLI configuration when environment
# graphs gain a dedicated typed construction API. It intentionally keeps graph
# parsing and construction out of the typed registered-environment path.
@dataclass
class LegacyCliEnvironmentCfg(ArenaEnvironmentCfg):
    """Carry arguments understood only by the legacy environment CLI adapter."""

    arguments: list[str]
    """Arguments passed to the existing environment parser."""

_BUILDER_FIELD_NAMES = {config_field.name for config_field in fields(ArenaEnvBuilderCfg)}
_LEGACY_EXPERIMENT_FIELD_NAMES = {
    "name",
    "arena_env_args",
    "policy_type",
    "policy_config_dict",
    "policy_args",
    "num_steps",
    "num_episodes",
    "num_rebuilds",
    "language_instruction",
    "variations",
    "status",
}


def arena_experiments_from_legacy_config(
    config: dict[str, Any],
    *,
    device: str,
) -> list[ArenaExperimentCfg]:
    """Translate the existing ``{"jobs": [...]}`` document into typed experiments."""
    assert set(config) == {"jobs"}, "legacy evaluation config must contain only a 'jobs' list"
    job_configs = config["jobs"]
    assert isinstance(job_configs, list), "legacy evaluation config 'jobs' must be a list"

    experiments = [_arena_experiment_from_legacy_job(job_config, device=device) for job_config in job_configs]
    experiment_names = [experiment.name for experiment in experiments]
    assert len(experiment_names) == len(set(experiment_names)), "experiment names must be unique"
    return experiments


def _arena_experiment_from_legacy_job(
    job_config: dict[str, Any],
    *,
    device: str,
) -> ArenaExperimentCfg:
    """Translate one legacy job mapping into a typed experiment."""
    assert isinstance(job_config, dict), "each legacy job must be a mapping"
    unknown_fields = set(job_config) - _LEGACY_EXPERIMENT_FIELD_NAMES
    assert not unknown_fields, f"unknown legacy job fields: {sorted(unknown_fields)}"
    for required_field in ("name", "arena_env_args", "policy_type"):
        assert required_field in job_config, f"{required_field} is required"

    status = job_config.get("status")
    assert status in (None, "pending"), "execution status does not belong in an experiment configuration"

    environment_cfg, builder_cfg = _environment_and_builder_cfgs_from_legacy_args(
        job_config["arena_env_args"],
        device=device,
        language_instruction=job_config.get("language_instruction"),
    )
    policy_cfg = _policy_cfg_from_legacy_job(job_config)

    variations = job_config.get("variations", {})
    assert isinstance(variations, dict), "variations must be a mapping"

    return ArenaExperimentCfg(
        name=job_config["name"],
        environment=environment_cfg,
        environment_builder=builder_cfg,
        policy=policy_cfg,
        rollout=RolloutCfg(
            num_steps=job_config.get("num_steps"),
            num_episodes=job_config.get("num_episodes"),
        ),
        num_rebuilds=job_config.get("num_rebuilds", 1),
        variations=variations,
    )


def _environment_and_builder_cfgs_from_legacy_args(
    arena_env_args: dict[str, Any],
    *,
    device: str,
    language_instruction: str | None,
) -> tuple[ArenaEnvironmentCfg, ArenaEnvBuilderCfg]:
    """Split the legacy mixed environment mapping into its two typed configs."""
    assert isinstance(arena_env_args, dict), "arena_env_args must be a mapping"
    assert arena_env_args.get("environment"), "arena_env_args.environment is required"

    environment_source = str(arena_env_args["environment"])
    builder_values = {
        field_name: value for field_name, value in arena_env_args.items() if field_name in _BUILDER_FIELD_NAMES
    }
    builder_values["device"] = device
    builder_values["language_instruction"] = language_instruction

    if environment_source.endswith((".yaml", ".yml")):
        environment_cfg = LegacyCliEnvironmentCfg(
            arguments=_legacy_graph_cli_args_from_dict(arena_env_args),
        )
        # The graph YAML declares its own dynamic override flags. The existing
        # parser remains responsible for validating them until graph construction
        # receives a typed API.
        accepted_fields = set(arena_env_args)
    else:
        ensure_environments_registered()
        environment_registry = EnvironmentRegistry()
        assert environment_registry.is_registered(
            environment_source, ensure_loaded=False
        ), f"Environment {environment_source!r} is not registered"
        environment_factory_type = environment_registry.get_component_by_name(environment_source)
        environment_cfg_type = environment_registry.get_environment_cfg_type(environment_factory_type)
        environment_field_names = {config_field.name for config_field in fields(environment_cfg_type)}
        environment_values = {
            field_name: value for field_name, value in arena_env_args.items() if field_name in environment_field_names
        }
        environment_cfg = environment_cfg_type(**environment_values)
        accepted_fields = _BUILDER_FIELD_NAMES | environment_field_names | {"environment"}

    unknown_fields = set(arena_env_args) - accepted_fields
    assert not unknown_fields, f"unknown arena_env_args fields for {environment_source!r}: {sorted(unknown_fields)}"
    return environment_cfg, ArenaEnvBuilderCfg(**builder_values)


# TODO(cvolk, 2026-07-06): Delete this graph-only dict-to-CLI conversion with
# LegacyCliEnvironmentCfg when graph construction accepts typed input.
def _legacy_graph_cli_args_from_dict(arena_env_args: dict[str, Any]) -> list[str]:
    """Reconstruct the existing graph CLI invocation from legacy JSON values."""
    priority_fields = ("num_envs", "env_spacing", "enable_cameras", "placement_seed")
    arguments: list[str] = []

    for field_name in priority_fields:
        if field_name not in arena_env_args:
            continue
        value = arena_env_args[field_name]
        if isinstance(value, bool):
            if value:
                arguments.append(f"--{field_name}")
        elif value is not None:
            arguments.extend((f"--{field_name}", str(value)))

    arguments.extend(("--env_graph_spec_yaml", str(arena_env_args["environment"])))
    for field_name, value in arena_env_args.items():
        if field_name in priority_fields or field_name == "environment":
            continue
        if isinstance(value, bool):
            if value:
                arguments.append(f"--{field_name}")
        elif value is not None:
            arguments.extend((f"--{field_name}", str(value)))
    return arguments


# TODO(cvolk, 2026-07-06): Delete this builder shim when environment graphs no
# longer require an argparse Namespace for construction.
def build_legacy_cli_arena_builder(experiment: ArenaExperimentCfg) -> ArenaEnvBuilder:
    """Build a graph experiment through the existing environment CLI frontend."""
    assert isinstance(experiment.environment, LegacyCliEnvironmentCfg)
    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(experiment.environment.arguments)
    args_cli.device = experiment.environment_builder.device
    args_cli.language_instruction = experiment.environment_builder.language_instruction
    return get_arena_builder_from_cli(
        args_cli,
        hydra_overrides=hydra_overrides_from_nested_dict(experiment.variations),
    )


def _policy_cfg_from_legacy_job(job_config: dict[str, Any]) -> PolicyCfg:
    """Construct a concrete registered policy config from legacy policy fields."""
    assert not (
        "policy_config_dict" in job_config and "policy_args" in job_config
    ), "provide only policy_config_dict; policy_args is a deprecated alias"
    policy_values = job_config.get("policy_config_dict", job_config.get("policy_args", {}))
    assert isinstance(policy_values, dict), "policy_config_dict must be a mapping"

    policy_registry = PolicyRegistry()
    policy_type = policy_registry.resolve_policy_type(job_config["policy_type"])
    policy_cfg_type = policy_registry.get_policy_cfg_type(policy_type)
    return policy_cfg_type(**policy_values)
