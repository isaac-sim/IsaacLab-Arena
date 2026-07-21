# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load typed Arena Experiments through Hydra."""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.policy_base import PolicyCfg


def _get_new_hydra_context_if_none_exists() -> AbstractContextManager[None]:
    """Initialize Hydra only when composition has no caller-owned context.

    Arena composes Experiments both standalone and from callers already using
    Hydra. Existing caller state must remain intact.
    """
    if GlobalHydra.instance().is_initialized():
        return nullcontext()
    return initialize(version_base=None, config_path=None)


def load_arena_experiment_from_yaml(
    yaml_path: str | Path,
    *,
    environment_cfg_types: dict[str, type[ArenaEnvironmentCfg]],
    policy_cfg_type_resolver: Callable[[str], type[PolicyCfg]],
    overrides: list[str] | None = None,
) -> ArenaExperimentCfg:
    """Load a YAML Arena Experiment Definition as a typed named-Run mapping.

    Each entry in the runs mapping declares one Run using its key as the Run
    name. The environment.type selector chooses from the supplied mapping,
    policy.type is resolved when its Run is built. Hydra overrides can update
    fields on Runs declared in YAML, but cannot add Runs.

    Args:
        yaml_path: Path to the Arena Experiment YAML file.
        environment_cfg_types: Environment selector names mapped to typed configuration classes.
        policy_cfg_type_resolver: Function returning the PolicyCfg subclass for a policy.type value.
        overrides: Hydra field overrides for Runs already declared in YAML.

    Returns:
        The typed Experiment Definition, preserving YAML mapping declaration order.
    """
    run_values_by_name = load_experiment_run_definitions_from_yaml(yaml_path)
    config_store = ConfigStore.instance()
    # Reuse these internal names so repeated loads replace their process-global ConfigStore entries.
    hydra_config_namespace = "isaaclab_arena_typed_experiment_loader"

    try:
        with _get_new_hydra_context_if_none_exists():
            arena_runs_by_name = {
                run_name: _build_arena_run_cfg_from_yaml_values(
                    config_store,
                    hydra_config_namespace,
                    index,
                    run_name,
                    run_values,
                    environment_cfg_types,
                    policy_cfg_type_resolver,
                )
                for index, (run_name, run_values) in enumerate(run_values_by_name.items())
            }
            hydra_experiment_config_name = f"{hydra_config_namespace}_root"
            config_store.store(
                name=hydra_experiment_config_name,
                node=ArenaExperimentCfg(runs=arena_runs_by_name),
            )
            composed_experiment = compose(config_name=hydra_experiment_config_name, overrides=overrides or [])
            experiment_cfg = OmegaConf.to_object(composed_experiment)
    except (HydraException, OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not compose Arena Experiment '{yaml_path}': {exc}") from exc

    assert isinstance(experiment_cfg, ArenaExperimentCfg)
    return experiment_cfg


def _assert_run_name_is_hydra_compatible(run_name: str) -> None:
    """Check that a Run name can be used directly in a Hydra override path."""
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", run_name), (
        f"Experiment Run name '{run_name}' must start with a letter or underscore and contain only letters, "
        "numbers, underscores, or hyphens"
    )


def load_experiment_run_definitions_from_yaml(yaml_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read an Arena Experiment YAML file and return its Run values by name.

    This validates the shared YAML envelope. Fields belonging to a Run,
    environment, or policy are validated later against their typed configs.

    Args:
        yaml_path: Path to the Arena Experiment YAML file.

    Returns:
        Run names mapped to their unresolved YAML values, in mapping declaration order.
    """
    path = Path(yaml_path)
    assert path.suffix.lower() in {".yaml", ".yml"}, f"Experiment config must be YAML, got '{path}'"
    assert path.is_file(), f"Experiment config does not exist: '{path}'"

    try:
        raw_experiment_config = OmegaConf.load(path)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Could not load Experiment YAML '{path}': {exc}") from exc
    assert OmegaConf.is_dict(raw_experiment_config), "Experiment config must be a mapping"

    unknown_fields = sorted(set(raw_experiment_config.keys()) - {"runs"})
    assert not unknown_fields, f"Unknown Experiment fields: {', '.join(unknown_fields)}"
    assert "runs" in raw_experiment_config, "Experiment config is missing the 'runs' field"
    assert OmegaConf.is_dict(
        raw_experiment_config.runs
    ), "Experiment 'runs' must be a mapping from Run names to Run configurations"
    assert raw_experiment_config.runs, "Experiment must define at least one Run"

    runs: dict[str, dict[str, Any]] = {}
    for run_name, raw_run_config in raw_experiment_config.runs.items():
        assert isinstance(run_name, str) and run_name, "Experiment Run names must be non-empty strings"
        _assert_run_name_is_hydra_compatible(run_name)
        assert OmegaConf.is_dict(raw_run_config), f"Run '{run_name}' must be a mapping"
        run_values = OmegaConf.to_container(raw_run_config, resolve=False)
        assert isinstance(run_values, dict)
        assert "name" not in run_values, f"Run '{run_name}' must not define 'name'; its mapping key is the Run name"
        runs[run_name] = run_values
    return runs


def _build_arena_run_cfg_from_yaml_values(
    config_store: ConfigStore,
    hydra_config_namespace: str,
    index: int,
    run_name: str,
    run_values: dict[str, Any],
    environment_cfg_types: dict[str, type[ArenaEnvironmentCfg]],
    policy_cfg_type_resolver: Callable[[str], type[PolicyCfg]],
) -> ArenaRunCfg:
    """Build one typed Arena Run from its unresolved YAML values.

    Args:
        config_store: Hydra store used for the temporary typed schemas.
        hydra_config_namespace: Unique prefix for this Experiment's temporary Hydra configs.
        index: Position of the Run in YAML declaration order.
        run_name: Name declared by the Run's YAML mapping key.
        run_values: Unresolved values declared for the Run.
        environment_cfg_types: Environment selectors mapped to typed configuration classes.
        policy_cfg_type_resolver: Function returning the PolicyCfg subclass for a policy.type value.

    Returns:
        The fully composed typed Run configuration.
    """
    remaining_values = dict(run_values)
    environment_values = remaining_values.pop("environment", None)
    policy_values = remaining_values.pop("policy", None)

    hydra_run_config_name = f"{hydra_config_namespace}_run_{index}"
    hydra_environment_config_name = f"{hydra_run_config_name}_environment"
    hydra_policy_config_name = f"{hydra_run_config_name}_policy"
    environment = _compose_typed_config_from_yaml_selector(
        config_store,
        hydra_environment_config_name,
        run_name,
        "environment",
        environment_values,
        environment_cfg_types,
        ArenaEnvironmentCfg,
    )
    policy_cfg_types: dict[str, type[PolicyCfg]] = {}
    if isinstance(policy_values, dict):
        policy_selector = policy_values.get("type")
        if isinstance(policy_selector, str) and policy_selector:
            policy_cfg_types[policy_selector] = policy_cfg_type_resolver(policy_selector)
    policy = _compose_typed_config_from_yaml_selector(
        config_store,
        hydra_policy_config_name,
        run_name,
        "policy",
        policy_values,
        policy_cfg_types,
        PolicyCfg,
    )

    hydra_run_schema_name = f"{hydra_run_config_name}_schema"
    config_store.store(name=hydra_run_schema_name, node=ArenaRunCfg(run_name, environment, policy))
    config_store.store(
        name=hydra_run_config_name,
        node={"defaults": [hydra_run_schema_name, "_self_"], **remaining_values},
    )
    run = OmegaConf.to_object(compose(config_name=hydra_run_config_name))
    assert isinstance(run, ArenaRunCfg)
    return run


def _compose_typed_config_from_yaml_selector(
    config_store: ConfigStore,
    hydra_config_name: str,
    run_name: str,
    section_name: str,
    section_values_with_selector: dict[str, Any],
    cfg_types: dict[str, type[Any]],
    expected_base_type: type[Any],
) -> Any:
    """Resolve one YAML type selector into a concrete typed configuration.

    The value under type selects a class from cfg_types. All remaining
    values are composed and validated against that class by Hydra.

    Args:
        config_store: Hydra store used for the temporary typed schema.
        hydra_config_name: Unique name for the temporary Hydra config.
        run_name: Run containing the selected environment or policy.
        section_name: YAML section being composed, such as environment or policy.
        section_values_with_selector: YAML field values for the section, including its type selector.
        cfg_types: Available selector names mapped to typed configuration classes.
        expected_base_type: Base class that the selected configuration must inherit.

    Returns:
        An instance of the selected concrete configuration class.
    """
    assert isinstance(section_values_with_selector, dict), f"Run '{run_name}' must define '{section_name}' as a mapping"
    values = section_values_with_selector.copy()
    selector = values.pop("type", None)
    assert isinstance(selector, str) and selector, f"Run '{run_name}' is missing '{section_name}.type'"
    available = ", ".join(sorted(cfg_types)) or "(none)"
    assert (
        selector in cfg_types
    ), f"Run '{run_name}' selects unknown {section_name} type '{selector}'. Available types: {available}"

    cfg_type = cfg_types[selector]
    assert isinstance(cfg_type, type) and issubclass(cfg_type, expected_base_type), (
        f"Configuration type registered for {section_name} selector '{selector}' "
        f"must inherit from {expected_base_type.__name__}"
    )

    hydra_schema_name = f"{hydra_config_name}_schema"
    config_store.store(name=hydra_schema_name, node=cfg_type)
    config_store.store(
        name=hydra_config_name,
        node={"defaults": [hydra_schema_name, "_self_"], **values},
    )
    return OmegaConf.to_object(compose(config_name=hydra_config_name))
