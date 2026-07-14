# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a complete Arena Experiment to OSMO."""

from __future__ import annotations

from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser

from isaaclab_arena.evaluation.experiment_runner_cfg import ExperimentRunnerCfg
from isaaclab_arena.hydra.experiment_composition import read_arena_experiment_run_values
from isaaclab_arena_openpi.policy.pi0_remote_config import DEFAULT_VARIANT
from osmo.workflows.arena_experiment_workflow import (
    ArenaExperimentWorkflow,
    OpenPiArenaExperimentWorkflow,
    load_arena_experiment_workflow_cfg,
)


def submit_experiment_to_osmo(
    experiment_runner_cfg: ExperimentRunnerCfg,
    osmo_config_path: str | None = None,
) -> int:
    """Build and submit one OSMO Workflow for a complete Arena Experiment.

    Args:
        experiment_runner_cfg: Final evaluation configuration, including trailing Experiment overrides.
        osmo_config_path: Optional path to the OSMO infrastructure configuration.

    Returns:
        The OSMO submission process status.
    """
    openpi_run_policy_values = _get_openpi_run_policy_values(experiment_runner_cfg.experiment_config)
    osmo_cfg = load_arena_experiment_workflow_cfg(osmo_config_path)

    if not openpi_run_policy_values:
        workflow = ArenaExperimentWorkflow(cfg=osmo_cfg, experiment_runner_cfg=experiment_runner_cfg)
        return workflow.submit_workflow()

    _assert_openpi_variants_match_server(
        experiment_runner_cfg,
        openpi_run_policy_values,
        configured_variant=osmo_cfg.openpi_server.policy_variant,
    )
    workflow = OpenPiArenaExperimentWorkflow(
        cfg=osmo_cfg,
        experiment_runner_cfg=experiment_runner_cfg,
        openpi_run_names=list(openpi_run_policy_values),
    )
    return workflow.submit_workflow()


def _get_openpi_run_policy_values(experiment_config_path: str) -> dict[str, dict[str, Any]]:
    """Return unresolved policy values for Runs selecting the OpenPI client."""
    openpi_run_policy_values = {}
    for run_name, run_values in read_arena_experiment_run_values(experiment_config_path).items():
        policy_values = run_values.get("policy")
        if isinstance(policy_values, dict) and policy_values.get("type") == "pi0_remote":
            openpi_run_policy_values[run_name] = policy_values
    return openpi_run_policy_values


def _assert_openpi_variants_match_server(
    experiment_runner_cfg: ExperimentRunnerCfg,
    openpi_run_policy_values: dict[str, dict[str, Any]],
    *,
    configured_variant: str,
) -> None:
    """Validate effective OpenPI Run variants against the shared OSMO server."""
    run_variants = {
        run_name: policy_values.get("policy_variant", DEFAULT_VARIANT)
        for run_name, policy_values in openpi_run_policy_values.items()
    }
    variant_paths = {f"runs.{run_name}.policy.policy_variant": run_name for run_name in openpi_run_policy_values}
    for override in OverridesParser.create().parse_overrides(experiment_runner_cfg.experiment_overrides):
        run_name = variant_paths.get(override.key_or_group)
        if run_name is not None:
            run_variants[run_name] = override.value()

    incompatible_runs = {
        run_name: variant for run_name, variant in run_variants.items() if variant != configured_variant
    }
    assert not incompatible_runs, (
        f"OpenPI Runs require variants {incompatible_runs}, but the OSMO server is configured for "
        f"'{configured_variant}'. One Arena Experiment currently supports one shared OpenPI server variant."
    )
