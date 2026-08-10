# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect one Arena Experiment result from its Run output directories."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from isaaclab_arena.visualization.episode_results_files import (
    find_episode_results_files,
    parse_episode_results_filename,
    read_episode_results,
)

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg

ARENA_EXPERIMENT_RESULT_FILENAME = "arena_experiment_result.json"


class ArenaEpisodeResultData(TypedDict):
    """Core data stored for one episode, with optional recorder data."""

    job_name: str
    env_id: int
    episode_in_env: int
    seed: int | None
    success: bool | None
    episode_length: int
    language_instruction: str | None
    timestamp: str
    variations: NotRequired[dict[str, Any]]
    progress: NotRequired[dict[str, Any]]


class ArenaRebuildResultData(TypedDict):
    """Episode results produced by one fresh environment construction."""

    index: int
    episodes: list[ArenaEpisodeResultData]


class ArenaExperimentRunData(TypedDict):
    """Data stored for one configured Run in an Arena Experiment result."""

    environment: dict[str, str]
    policy_variant: str
    status: Literal["completed", "failed"]
    rebuilds: list[ArenaRebuildResultData]


ArenaExperimentResultData = dict[Literal["runs"], dict[str, ArenaExperimentRunData]]


def _require_nonempty_string(value: object, field_name: str) -> str:
    """Return a non-empty string used in Run metadata."""
    assert isinstance(value, str) and value.strip(), f"{field_name} must be a non-empty string"
    return value


def _collect_rebuilds(
    run_output_directory: Path,
    experiment_output_directory: Path,
) -> list[ArenaRebuildResultData]:
    """Collect raw episode objects from one Run directory, grouped by rebuild."""
    episodes_by_rebuild: dict[int, list[ArenaEpisodeResultData]] = defaultdict(list)
    for episode_results_path in find_episode_results_files(run_output_directory):
        parsed_filename = parse_episode_results_filename(episode_results_path.name)
        assert parsed_filename is not None, f"episode results filename did not parse: {episode_results_path.name}"
        episodes_by_rebuild[parsed_filename.rebuild_index]
        episode_results, data_issues = read_episode_results(
            episode_results_path,
            experiment_output_directory,
        )
        assert not data_issues, "; ".join(f"{data_issue.path}: {data_issue.message}" for data_issue in data_issues)
        episodes_by_rebuild[parsed_filename.rebuild_index].extend(episode_results)

    return [
        {"index": rebuild_index, "episodes": episodes_by_rebuild[rebuild_index]}
        for rebuild_index in sorted(episodes_by_rebuild)
    ]


class ArenaExperimentResult:
    """Combine every Run's metadata and episode JSONLs into one Experiment result file.

    The result contains each configured Run's environment, policy variant, execution status,
    and raw episode records grouped by rebuild. It does not compute aggregate metrics or collect videos.
    """

    @staticmethod
    def assert_run_name_is_safe_path_component(run_name: str) -> None:
        """Check that a Run name selects one direct child of an output directory."""
        assert isinstance(run_name, str) and run_name.strip(), "Run name must be a non-empty string"
        assert (
            run_name not in {".", ".."} and "/" not in run_name and "\\" not in run_name
        ), f"Run name {run_name!r} must be one safe path component"
        assert not PureWindowsPath(run_name).drive, f"Run name {run_name!r} must not contain a drive"

    def __init__(
        self,
        experiment_output_directory: str | Path,
        run_metadata_by_name: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Collect declared Run outputs from an Experiment output directory.

        Args:
            experiment_output_directory: Experiment output root containing one directory per Run.
            run_metadata_by_name: Environment, policy variant, and final status for each declared Run.
        """
        assert run_metadata_by_name, "run_metadata_by_name must not be empty"
        self.experiment_output_directory = Path(experiment_output_directory)
        self._runs: dict[str, ArenaExperimentRunData] = {
            run_name: self._collect_run(run_name, run_metadata)
            for run_name, run_metadata in run_metadata_by_name.items()
        }

    def _collect_run(self, run_name: str, run_metadata: Mapping[str, Any]) -> ArenaExperimentRunData:
        """Collect one declared Run's metadata and episode results."""
        self.assert_run_name_is_safe_path_component(run_name)
        environment = run_metadata["environment"]
        status = run_metadata["status"]
        assert status in ("completed", "failed"), f"status for Run {run_name!r} must be 'completed' or 'failed'"

        run_output_directory = self.experiment_output_directory / run_name
        if status == "completed":
            assert (
                run_output_directory.is_dir()
            ), f"completed Run {run_name!r} is missing its output directory: {run_output_directory}"
        elif run_output_directory.exists():
            assert (
                run_output_directory.is_dir()
            ), f"output path for failed Run {run_name!r} must be a directory: {run_output_directory}"

        rebuilds = (
            _collect_rebuilds(run_output_directory, self.experiment_output_directory)
            if run_output_directory.is_dir()
            else []
        )
        return {
            "environment": {
                "name": environment["name"],
                "definition": environment["definition"],
            },
            "policy_variant": run_metadata["policy_variant"],
            "status": status,
            "rebuilds": rebuilds,
        }

    def to_dict(self) -> ArenaExperimentResultData:
        """Return the collected JSON-compatible Experiment result."""
        return {"runs": self._runs}

    def write(self) -> Path:
        """Write all collected Runs to ``arena_experiment_result.json`` in the Experiment output directory."""
        result_path = self.experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME
        result_path.write_text(
            json.dumps(self.to_dict(), allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote Arena Experiment result to: {result_path}")
        return result_path


def build_arena_run_result_metadata(run_cfg: ArenaRunCfg) -> dict[str, Any]:
    """Build serializable environment and policy metadata for an effective Run."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry

    environment_cfg = run_cfg.environment
    # TODO(cvolk): Remove this structural check once graph-YAML environments use registered typed configurations.
    if hasattr(environment_cfg, "env_spec_path"):
        from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict

        environment_definition = _require_nonempty_string(
            environment_cfg.env_spec_path,
            "graph environment definition",
        )
        environment_graph = load_env_graph_spec_dict(environment_definition)
        environment_name = _require_nonempty_string(
            environment_graph.get("env_name"),
            f"env_name in graph environment definition {environment_definition!r}",
        )
    else:
        environment_factory_type = EnvironmentRegistry().get_factory_type_for_cfg(environment_cfg)
        environment_name = _require_nonempty_string(
            environment_factory_type.name,
            f"registered environment name for {type(environment_cfg).__name__}",
        )
        environment_definition = environment_name

    configured_policy_variant = getattr(run_cfg.policy, "policy_variant", None)
    if isinstance(configured_policy_variant, str) and configured_policy_variant.strip():
        policy_variant = configured_policy_variant
    else:
        policy_type = PolicyRegistry().get_policy_type_for_cfg(run_cfg.policy)
        policy_variant = _require_nonempty_string(
            policy_type.name,
            f"registered policy name for {type(run_cfg.policy).__name__}",
        )

    return {
        "environment": {
            "name": environment_name,
            "definition": environment_definition,
        },
        "policy_variant": policy_variant,
    }
