# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate typed Arena experiment collections."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.arena_experiment import (
    ArenaExperimentCollectionCfg,
    ArenaExperimentResult,
    ExperimentStatus,
)
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.experiment_collection_hydra import compose_experiment_collection
from isaaclab_arena.evaluation.experiment_execution import build_and_run_experiment, build_arena_builder_for_experiment
from isaaclab_arena.evaluation.legacy_eval_config import experiment_collection_from_legacy_eval_config
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

# TODO(cvolk, 2026-07-07): Replace these first migrated config types with an
# extension-owned lightweight registration mechanism as more environment and policy
# configurations move to YAML. Importing every runtime implementation before
# SimulationApp would also import pxr/PhysX modules too early.
_YAML_ENVIRONMENT_CFG_TYPES = {
    "pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg,
}
_YAML_POLICY_CFG_TYPES = {
    "zero_action": ZeroActionPolicyCfg,
}


def _resolve_config_path(args_cli: argparse.Namespace) -> Path:
    """Return the one experiment configuration selected on the command line."""
    positional_path = args_cli.experiment_config
    legacy_path = args_cli.legacy_eval_jobs_config
    assert (positional_path is None) != (
        legacy_path is None
    ), "Provide exactly one experiment configuration: a positional YAML path or --eval_jobs_config for legacy JSON"
    path = Path(positional_path or legacy_path).resolve()
    assert path.is_file(), f"experiment configuration does not exist: {path}"
    if legacy_path is not None:
        assert path.suffix == ".json", "--eval_jobs_config only accepts legacy JSON configurations"
    else:
        assert path.suffix in (".yaml", ".yml"), "the positional experiment configuration must be YAML"
    return path


def _compose_typed_experiment_collection(
    config_path: Path,
    device: str,
    hydra_overrides: list[str],
) -> ArenaExperimentCollectionCfg:
    """Compose the typed YAML frontend and apply its process-wide device."""
    collection_cfg = compose_experiment_collection(
        config_path,
        environment_cfg_types=_YAML_ENVIRONMENT_CFG_TYPES,
        policy_cfg_types=_YAML_POLICY_CFG_TYPES,
        overrides=hydra_overrides,
    )
    return _collection_with_device(collection_cfg, device)


def _collection_with_device(
    collection_cfg: ArenaExperimentCollectionCfg,
    device: str,
) -> ArenaExperimentCollectionCfg:
    """Apply the process-wide simulation device to every experiment builder."""
    return ArenaExperimentCollectionCfg(
        experiments={
            experiment_name: replace(
                experiment_cfg,
                environment_builder=replace(experiment_cfg.environment_builder, device=device),
            )
            for experiment_name, experiment_cfg in collection_cfg.experiments.items()
        }
    )


def _typed_collection_enables_cameras(collection_cfg: ArenaExperimentCollectionCfg) -> bool:
    """Return whether any typed experiment enables environment cameras."""
    return any(
        _environment_enables_cameras(experiment_cfg.environment)
        for experiment_cfg in collection_cfg.experiments.values()
    )


# TODO(cvolk, 2026-07-07): Delete the JSON loading and camera-inspection helpers
# below with --eval_jobs_config after the remaining JSON documents migrate to typed
# YAML collections.
def _load_legacy_eval_config(config_path: Path, hydra_overrides: list[str]) -> dict[str, Any]:
    """Load the retained JSON document without importing its runtime implementations."""
    assert not hydra_overrides, "Hydra overrides are supported only for typed YAML experiment collections"
    with config_path.open(encoding="utf-8") as config_file:
        legacy_config = json.load(config_file)
    assert isinstance(legacy_config, dict), "legacy evaluation configuration must be a mapping"
    return legacy_config


def _legacy_eval_config_enables_cameras(legacy_config: dict[str, Any]) -> bool:
    """Read camera startup requirements without adapting legacy JSON before SimulationApp."""
    jobs = legacy_config.get("jobs", [])
    assert isinstance(jobs, list), "legacy evaluation config 'jobs' must be a list"
    return any(
        isinstance(job_config, dict)
        and isinstance(job_config.get("arena_env_args"), dict)
        and job_config["arena_env_args"].get("enable_cameras", False)
        for job_config in jobs
    )


def _environment_enables_cameras(environment_cfg: object) -> bool:
    """Return whether one concrete typed environment enables cameras."""
    return bool(getattr(environment_cfg, "enable_cameras", False))


def _list_variations(collection_cfg: ArenaExperimentCollectionCfg) -> None:
    """Print the Hydra-configurable variations for every experiment."""
    for experiment_name, experiment_cfg in collection_cfg.experiments.items():
        arena_builder = build_arena_builder_for_experiment(experiment_cfg)
        print(f"=== Variations for experiment '{experiment_name}' ===", flush=True)
        print(arena_builder.get_variations_catalogue_as_string(), flush=True)


def _run_experiments(
    collection_cfg: ArenaExperimentCollectionCfg,
    args_cli: argparse.Namespace,
) -> dict[str, ArenaExperimentResult]:
    """Run the configured experiments sequentially and return their results."""
    metrics_logger = MetricsLogger()
    results = {}
    run_output_dir = timestamped_run_dir(args_cli.output_base_dir)

    if args_cli.record_viewport_video:
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"[INFO] Video recording enabled. Videos will be saved to: {run_output_dir}")

    for experiment_name, experiment_cfg in collection_cfg.experiments.items():
        print(f"Running experiment {experiment_name}", flush=True)
        experiment_output_dir = os.path.join(run_output_dir, experiment_name)
        try:
            result = build_and_run_experiment(
                experiment_cfg,
                output_dir=experiment_output_dir,
                video_cfg=VideoRecordingCfg(
                    record_viewport_video=args_cli.record_viewport_video,
                    record_camera_video=args_cli.record_camera_video,
                    video_base_dir=experiment_output_dir,
                ),
            )
        except Exception as error:
            result = ArenaExperimentResult(
                experiment_name=experiment_name,
                status=ExperimentStatus.FAILED,
            )
            results[experiment_name] = result
            print(f"Experiment {experiment_name} failed with error: {error}")
            print(f"Traceback: {traceback.format_exc()}")
            if not args_cli.continue_on_error:
                raise
            continue

        results[experiment_name] = result
        if result.metrics is not None:
            metrics_logger.append_job_metrics(experiment_name, result.metrics)

    _print_experiment_results(collection_cfg, results)
    metrics_logger.print_metrics()

    report_path = build_report(run_output_dir)
    if args_cli.serve_evaluation_report:
        serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)
    return results


def _print_experiment_results(
    collection_cfg: ArenaExperimentCollectionCfg,
    results: dict[str, ArenaExperimentResult],
) -> None:
    """Print one final status line for every configured experiment."""
    print("Experiment results:", flush=True)
    for experiment_name in collection_cfg.experiments:
        result = results.get(experiment_name)
        status = result.status.value if result is not None else "not run"
        print(f"  {experiment_name}: {status}", flush=True)


def main() -> None:
    """Load and evaluate one typed or legacy experiment collection."""
    args_parser = get_isaaclab_arena_cli_parser()
    add_eval_runner_arguments(args_parser)
    args_cli, hydra_overrides = args_parser.parse_known_args()
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"

    config_path = _resolve_config_path(args_cli)
    if config_path.suffix in (".yaml", ".yml"):
        assert_hydra_overrides(hydra_overrides, args_parser)
        collection_cfg = _compose_typed_experiment_collection(config_path, args_cli.device, hydra_overrides)
        legacy_config = None
        requires_environment_cameras = _typed_collection_enables_cameras(collection_cfg)
    else:
        legacy_config = _load_legacy_eval_config(config_path, hydra_overrides)
        collection_cfg = None
        requires_environment_cameras = _legacy_eval_config_enables_cameras(legacy_config)
    args_cli.enable_cameras = bool(
        args_cli.enable_cameras or args_cli.record_camera_video or requires_environment_cameras
    )

    with SimulationAppContext(args_cli):
        # TODO(cvolk, 2026-07-07): Remove this in-simulator JSON adaptation after
        # every evaluation configuration uses the typed YAML frontend. Importing all
        # legacy implementations before SimulationApp would load pxr/PhysX too early.
        if collection_cfg is None:
            assert legacy_config is not None
            collection_cfg = experiment_collection_from_legacy_eval_config(legacy_config, device=args_cli.device)
        if args_cli.list_variations:
            _list_variations(collection_cfg)
            return
        _run_experiments(collection_cfg, args_cli)


if __name__ == "__main__":
    main()
