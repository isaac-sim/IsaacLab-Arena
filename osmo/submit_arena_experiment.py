# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose and submit one Arena Experiment to OSMO.

Experiment Definitions must be direct ``.yaml`` or ``.yml`` files in
``isaaclab_arena_environments/experiment_configs``. Select one by filename
stem with ``experiment_definition=<name>``; for example,
``experiment_definition=openpi_experiment`` selects
``isaaclab_arena_environments/experiment_configs/openpi_experiment.yaml``.
Arbitrary Experiment Definition paths are not supported.

Example:

    python -m osmo.submit_arena_experiment \
        experiment_definition=openpi_experiment \
        server_config=pi0 \
        osmo_config.pool=isaac-dev-l40-03 \
        osmo_config.platform=ovx-l40 \
        osmo_config.memory=120Gi \
        osmo_config.workflow_name=my-evaluation \
        experiment_runner_config.image=nvcr.io/example/isaaclab_arena:experiment_runner \
        experiment_definition.runs.openpi_maple_table.rollout_limit.num_episodes=4

The named config groups select an Experiment Definition, optional policy-server
definition, OSMO scheduling settings, and Experiment Runner task settings. Hydra
applies trailing field overrides after the selected files.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from osmo.arena_experiment_submission import (
    POLICY_SERVER_WORKFLOWS,
    ArenaExperimentSubmissionCfg,
    submit_arena_experiment,
)

CONFIG_DIR = Path(__file__).parent / "config"
SUBMISSION_CONFIG_NAME = "arena_experiment_submission"
SUBMISSION_SCHEMA_NAME = "arena_experiment_submission_schema"


def compose_arena_experiment_submission(overrides: list[str] | None = None) -> ArenaExperimentSubmissionCfg:
    """Compose a typed submission from named config groups and Hydra overrides.

    Args:
        overrides: Hydra config-group selections and field overrides.

    Returns:
        The fully composed typed submission configuration.
    """
    _register_hydra_configs()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        composed = compose(config_name=SUBMISSION_CONFIG_NAME, overrides=overrides or [])
    return _submission_cfg_from_hydra(composed)


def _submission_cfg_from_hydra(composed: Any) -> ArenaExperimentSubmissionCfg:
    """Convert Hydra's composed object into the typed submission root."""
    submission_cfg = OmegaConf.to_object(composed)
    assert isinstance(submission_cfg, ArenaExperimentSubmissionCfg)
    return submission_cfg


@cache
def _register_hydra_configs() -> None:
    """Register the submission schema and named config-group choices."""
    config_store = ConfigStore.instance()
    config_store.store(name=SUBMISSION_SCHEMA_NAME, node=ArenaExperimentSubmissionCfg)
    for server_name, workflow_cls in POLICY_SERVER_WORKFLOWS.items():
        config_store.store(group="server_config", name=server_name, node=workflow_cls.server_task_cfg_type)

    import isaaclab_arena_environments

    experiment_definition_dir = Path(isaaclab_arena_environments.__file__).parent / "experiment_configs"
    config_paths = sorted([*experiment_definition_dir.glob("*.yaml"), *experiment_definition_dir.glob("*.yml")])
    assert config_paths, f"No Arena Experiment Definitions found in '{experiment_definition_dir}'"
    for config_path in config_paths:
        experiment_definition = load_arena_experiment_from_config_file(config_path, device="cuda:0")
        config_store.store(
            group="experiment_definition",
            name=config_path.stem,
            node=experiment_definition,
        )


@hydra_main(version_base=None, config_path="config", config_name=SUBMISSION_CONFIG_NAME)
def _submit_arena_experiment_from_hydra(composed: Any) -> None:
    """Submit the Arena Experiment composed by Hydra's command-line frontend."""
    status = submit_arena_experiment(_submission_cfg_from_hydra(composed))
    if status:
        raise SystemExit(status)


def main() -> int:
    """Run the Hydra submission CLI."""
    _register_hydra_configs()
    _submit_arena_experiment_from_hydra()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
