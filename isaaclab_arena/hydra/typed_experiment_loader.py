# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build typed Arena Experiments through Hydra from the values of an Experiment."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
    from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
    from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
    from isaaclab_arena.policy.policy_base import PolicyCfg


def _get_new_hydra_context_if_none_exists() -> AbstractContextManager[None]:
    """Initialize Hydra only when composition has no caller-owned context.

    Arena composes Experiments both standalone and from callers already using
    Hydra. Existing caller state must remain intact.
    """
    if GlobalHydra.instance().is_initialized():
        return nullcontext()
    return initialize(version_base=None, config_path=None)


def build_typed_experiment(
    experiment_values: dict[str, Any],
    *,
    environment_cfg_types: dict[str, type[ArenaEnvironmentCfg]],
    policy_cfg_type_resolver: Callable[[str], type[PolicyCfg]],
) -> ArenaExperimentCfg:
    """Build an Arena Experiment Definition with typed named Runs from the Experiment's values.

    Each Run's environment.type and policy.type select its typed configs. environment.type may also name a
    graph-spec YAML path.

    Args:
        experiment_values: The values returned by read_experiment_values, with ``shared`` merged into each Run.
        environment_cfg_types: Environment selector names mapped to typed configuration classes.
        policy_cfg_type_resolver: Function returning the PolicyCfg subclass for a policy.type value.

    Returns:
        The typed Experiment Definition, preserving Run declaration order.
    """
    from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg

    # An Experiment without kpi_exporters keeps the ArenaExperimentCfg default.
    kpi_exporters_kwargs = {}
    if "kpi_exporters" in experiment_values:
        kpi_exporters_kwargs["kpi_exporters"] = experiment_values["kpi_exporters"]
    config_store = ConfigStore.instance()
    # Reuse these internal names so repeated loads replace their process-global ConfigStore entries.
    hydra_config_namespace = "isaaclab_arena_typed_experiment_loader"

    arena_runs_by_name = {}
    with _get_new_hydra_context_if_none_exists():
        for index, (run_name, run_values) in enumerate(experiment_values["runs"].items()):
            try:
                arena_runs_by_name[run_name] = _build_arena_run_cfg(
                    config_store,
                    hydra_config_namespace,
                    index,
                    run_name,
                    run_values,
                    environment_cfg_types,
                    policy_cfg_type_resolver,
                )
            except AssertionError as exc:
                raise AssertionError(f"Run '{run_name}': {exc}") from exc
            except (HydraException, OmegaConfBaseException, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not compose Arena Experiment '{experiment_values['name']}': Run '{run_name}': {exc}"
                ) from exc
    return ArenaExperimentCfg(runs=arena_runs_by_name, name=experiment_values["name"], **kpi_exporters_kwargs)


def _build_arena_run_cfg(
    config_store: ConfigStore,
    hydra_config_namespace: str,
    index: int,
    run_name: str,
    run_values: dict[str, Any],
    environment_cfg_types: dict[str, type[ArenaEnvironmentCfg]],
    policy_cfg_type_resolver: Callable[[str], type[PolicyCfg]],
) -> ArenaRunCfg:
    """Build one typed Arena Run from its unresolved values.

    Args:
        config_store: Hydra store used for the temporary typed schemas.
        hydra_config_namespace: Unique prefix for this Experiment's temporary Hydra configs.
        index: Position of the Run in declaration order.
        run_name: Name of the Run, its key under ``runs``.
        run_values: Values declared for the Run after applying shared defaults.
        environment_cfg_types: Environment selectors mapped to typed configuration classes.
        policy_cfg_type_resolver: Function returning the PolicyCfg subclass for a policy.type value.

    Returns:
        The fully composed typed Run configuration.
    """
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
    from isaaclab_arena.policy.policy_base import PolicyCfg

    remaining_values = dict(run_values)
    environment_values = remaining_values.pop("environment")
    policy_values = remaining_values.pop("policy")

    hydra_run_config_name = f"{hydra_config_namespace}_run_{index}"
    hydra_environment_config_name = f"{hydra_run_config_name}_environment"
    hydra_policy_config_name = f"{hydra_run_config_name}_policy"
    environment = _build_environment_cfg(
        config_store,
        hydra_environment_config_name,
        environment_values,
        environment_cfg_types,
    )
    policy = _compose_typed_config(
        config_store,
        hydra_policy_config_name,
        "policy",
        policy_values,
        policy_cfg_type_resolver(policy_values["type"]),
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


def _build_environment_cfg(
    config_store: ConfigStore,
    hydra_environment_config_name: str,
    environment_values: dict[str, Any],
    environment_cfg_types: dict[str, type[ArenaEnvironmentCfg]],
) -> ArenaEnvironmentCfg:
    """Build a Run's environment from a graph-spec YAML path or a typed selector.

    When environment.type names a graph-spec YAML file it is built on the temporary
    argparse compatibility path; otherwise the type selects a registered typed config.
    """
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

    environment_type = environment_values["type"]
    if environment_type.lower().endswith((".yaml", ".yml")):
        per_run_overrides = {
            field_name: value for field_name, value in environment_values.items() if field_name != "type"
        }
        return _build_graph_environment_cfg(environment_type, per_run_overrides)
    available_types = ", ".join(sorted(environment_cfg_types)) or "(none)"
    assert (
        environment_type in environment_cfg_types
    ), f"unknown environment type '{environment_type}'. Available types: {available_types}"
    return _compose_typed_config(
        config_store,
        hydra_environment_config_name,
        "environment",
        environment_values,
        environment_cfg_types[environment_type],
        ArenaEnvironmentCfg,
    )


# TODO(cvolk, 2026-07-07): [typed-config-migration] Delete this factory when graph-YAML
# environments have a typed configuration and no longer use the argparse compatibility path.
def _build_graph_environment_cfg(
    env_spec_path: str,
    per_run_overrides: dict[str, Any],
) -> LegacyGraphEnvironmentCfg:
    """Create the temporary graph-YAML compatibility config from a Run's environment values.

    The path and environment values are stored structured and rendered into CLI tokens for
    the existing graph-environment argparse path on demand at execution; the Run's
    environment_builder section stays typed and is applied directly (see
    build_arena_builder_from_legacy_graph).
    """
    from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg

    return LegacyGraphEnvironmentCfg(
        enable_cameras=bool(per_run_overrides.get("enable_cameras", False)),
        env_spec_path=env_spec_path,
        per_run_overrides=dict(per_run_overrides),
    )


def _compose_typed_config(
    config_store: ConfigStore,
    hydra_config_name: str,
    section_name: str,
    section_values: dict[str, Any],
    cfg_type: type[Any],
    expected_base_type: type[Any],
) -> Any:
    """Compose one Run section into an instance of the typed configuration class its ``type`` selects.

    Hydra validates every value of the section except ``type`` against ``cfg_type``.

    Args:
        config_store: Hydra store used for the temporary typed schema.
        hydra_config_name: Unique name for the temporary Hydra config.
        section_name: Section being composed: environment or policy.
        section_values: Field values of the section, including its ``type``.
        cfg_type: Typed configuration class selected by the section's ``type``.
        expected_base_type: Base class that the selected configuration must inherit.

    Returns:
        An instance of ``cfg_type``.
    """
    values = dict(section_values)
    selector = values.pop("type")
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
