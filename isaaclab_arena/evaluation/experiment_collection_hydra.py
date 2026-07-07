# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose typed Arena experiment collections from keyed YAML configurations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg, ArenaExperimentCollectionCfg
from isaaclab_arena.policy.policy_base import PolicyCfg

CfgT = TypeVar("CfgT")


def compose_experiment_collection(
    config_path: str | Path,
    environment_cfg_types: Mapping[str, type[ArenaEnvironmentCfg]],
    policy_cfg_types: Mapping[str, type[PolicyCfg]],
    overrides: list[str] | None = None,
) -> ArenaExperimentCollectionCfg:
    """Compose a keyed YAML document into concrete experiment configs.

    Args:
        config_path: YAML file containing the experiment collection.
        environment_cfg_types: Concrete environment configs available to YAML selectors.
        policy_cfg_types: Concrete policy configs available to YAML selectors.
        overrides: Hydra value overrides applied after the YAML values.

    Returns:
        The fully typed experiment collection in YAML declaration order.
    """
    path = Path(config_path).resolve()
    assert path.is_file(), f"experiment configuration does not exist: {path}"
    assert path.suffix in (".yaml", ".yml"), f"experiment configuration must be YAML: {path}"

    document = OmegaConf.load(path)
    assert isinstance(document, DictConfig), "experiment configuration must be a mapping"
    assert set(document) == {"experiments"}, "experiment configuration must contain only 'experiments'"
    experiment_nodes = document.experiments
    assert isinstance(experiment_nodes, DictConfig), "experiments must be a mapping keyed by experiment name"
    assert experiment_nodes, "experiment collection must not be empty"

    config_store = ConfigStore.instance()
    config_namespace = f"_arena_experiment_collection_{uuid4().hex}"
    experiment_schema_name = f"{config_namespace}_experiment_schema"
    environment_group = f"{config_namespace}_environment"
    policy_group = f"{config_namespace}_policy"
    config_store.store(name=experiment_schema_name, node=ArenaExperimentCfg)

    registered_environment_types: set[str] = set()
    registered_policy_types: set[str] = set()
    experiment_config_names: list[str] = []

    for experiment_name, experiment_node in experiment_nodes.items():
        assert isinstance(experiment_name, str) and experiment_name, "experiment names must not be empty"
        assert isinstance(experiment_node, DictConfig), f"experiment '{experiment_name}' must be a mapping"
        assert (
            "name" not in experiment_node
        ), f"experiment '{experiment_name}' must use its mapping key as its name; remove the nested name field"

        experiment_values = OmegaConf.to_container(experiment_node, resolve=False)
        assert isinstance(experiment_values, dict)
        environment_values, environment_type_name = _pop_config_selector(
            experiment_values,
            field_name="environment",
            experiment_name=experiment_name,
        )
        policy_values, policy_type_name = _pop_config_selector(
            experiment_values,
            field_name="policy",
            experiment_name=experiment_name,
        )

        if environment_type_name not in registered_environment_types:
            config_store.store(
                group=environment_group,
                name=environment_type_name,
                node=_selected_cfg_type(environment_type_name, environment_cfg_types, "environment"),
            )
            registered_environment_types.add(environment_type_name)
        if policy_type_name not in registered_policy_types:
            config_store.store(
                group=policy_group,
                name=policy_type_name,
                node=_selected_cfg_type(policy_type_name, policy_cfg_types, "policy"),
            )
            registered_policy_types.add(policy_type_name)

        experiment_config_name = f"{config_namespace}_experiment_{len(experiment_config_names)}"
        config_store.store(
            name=experiment_config_name,
            node={
                "defaults": [
                    experiment_schema_name,
                    {f"{environment_group}@environment": environment_type_name},
                    {f"{policy_group}@policy": policy_type_name},
                    "_self_",
                ],
                "name": experiment_name,
                **experiment_values,
                "environment": environment_values,
                "policy": policy_values,
            },
        )
        experiment_config_names.append(experiment_config_name)

    with initialize(version_base=None, config_path=None):
        experiments = {}
        for experiment_name, experiment_config_name in zip(experiment_nodes, experiment_config_names, strict=True):
            experiment_cfg = OmegaConf.to_object(compose(config_name=experiment_config_name))
            assert isinstance(experiment_cfg, ArenaExperimentCfg)
            experiments[experiment_name] = experiment_cfg

        collection_config_name = f"{config_namespace}_collection"
        config_store.store(
            name=collection_config_name,
            node=ArenaExperimentCollectionCfg(experiments=experiments),
        )
        collection_cfg = OmegaConf.to_object(
            compose(
                config_name=collection_config_name,
                overrides=list(overrides or []),
            )
        )

    assert isinstance(collection_cfg, ArenaExperimentCollectionCfg)
    return collection_cfg


def _pop_config_selector(
    experiment_values: dict[str, Any],
    field_name: str,
    experiment_name: str,
) -> tuple[dict[str, Any], str]:
    """Remove one YAML ``type`` selector and return its remaining values."""
    field_values = experiment_values.get(field_name)
    assert isinstance(field_values, dict), f"experiment '{experiment_name}'.{field_name} must be a mapping"
    type_name = field_values.pop("type", None)
    assert isinstance(type_name, str) and type_name, f"experiment '{experiment_name}'.{field_name}.type is required"
    return field_values, type_name


def _selected_cfg_type(
    type_name: str,
    cfg_types: Mapping[str, type[CfgT]],
    field_name: str,
) -> type[CfgT]:
    """Resolve one YAML selector from the config types provided by the frontend."""
    assert type_name in cfg_types, f"{field_name} type '{type_name}' is not available for typed YAML composition"
    return cfg_types[type_name]
