# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect, write and read the Arena Experiment result."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from isaaclab_arena.evaluation.run_status import RunStatus
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena.utils.string_checks import is_single_line_nonempty_string
from isaaclab_arena.visualization.episode_results_files import (
    find_episode_results_files,
    parse_episode_results_filename,
    read_episode_results,
)

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg

ARENA_EXPERIMENT_RESULT_FILENAME = "arena_experiment_result.json"
ARENA_SINGLE_RUN_TIMINGS_FILENAME = "arena_run_timings.json"
ARENA_AGGREGATED_TIMINGS_FILENAME = "arena_experiment_timings.json"
ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME = "arena_run_placement_diagnostics.json"
ARENA_SINGLE_RUN_INFERENCE_USAGE_FILENAME = "arena_run_inference_usage.json"
"""Name of a Run's inference usage file, which lists the inference usage of each rebuild's policy in order."""


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
    """Maps ``name`` and ``definition`` to the Run's environment name and definition."""

    policy_variant: str
    status: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    rebuilds: list[ArenaRebuildResultData]


class ArenaExperimentResultData(TypedDict):
    """Contents of an Arena Experiment result file."""

    git_stamp: dict[str, str]
    """JSON object of the Arena ``GitStamp``."""

    experiment_name: str
    """Name of the Experiment, a non-empty single-line string."""

    runs: dict[str, ArenaExperimentRunData]
    """Maps each configured Run name to that Run's data."""


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
    """One Arena Experiment result: its name, Arena git stamp and the data of every configured Run.

    Each Run holds its environment, policy variant, execution status, start and end times, and raw episode records
    grouped by rebuild. The result does not compute aggregate metrics or collect videos.
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
        experiment_name: str,
        git_stamp: GitStamp,
        runs: Mapping[str, ArenaExperimentRunData],
    ) -> None:
        """Construct a result from already collected Run data.

        Intended for internal use; prefer ``from_experiment_output`` to load a written result or
        ``from_run_outputs`` to collect the outputs of executed Runs.

        Args:
            experiment_output_directory: Experiment output root the result file is written to.
            experiment_name: Name of the Experiment, a non-empty single-line string.
            git_stamp: Arena git commit and branch the Experiment ran with.
            runs: Data of each configured Run by Run name, with an integer end time for every Run.
        """
        assert is_single_line_nonempty_string(
            experiment_name
        ), f"experiment_name must be a non-empty single-line string, got {experiment_name!r}"
        assert runs, "runs must not be empty"
        for run_name, run_data in runs.items():
            end_timestamp_ns = run_data.get("end_timestamp_ns")
            assert isinstance(end_timestamp_ns, int) and not isinstance(end_timestamp_ns, bool), (
                f"Run {run_name!r} in '{experiment_output_directory}' has no integer end_timestamp_ns; results"
                " written before Run end times were recorded are not supported"
            )
        self.experiment_output_directory = Path(experiment_output_directory)
        self.experiment_name = experiment_name
        self.git_stamp = git_stamp
        self.runs: dict[str, ArenaExperimentRunData] = dict(runs)

    @classmethod
    def from_experiment_output(cls, output_dir: str | Path) -> ArenaExperimentResult:
        """Load and check ``arena_experiment_result.json`` of an Experiment output directory.

        Args:
            output_dir: Experiment output root holding the result file.

        Returns:
            The Experiment result, with at least one Run, a git stamp, an Experiment name and an integer end time for
            every Run.
        """
        output_dir = Path(output_dir)
        data: ArenaExperimentResultData = json.loads(
            (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8")
        )
        assert "git_stamp" in data, f"Experiment result in '{output_dir}' has no git stamp"
        assert "experiment_name" in data, f"Experiment result in '{output_dir}' has no experiment_name"
        assert "runs" in data, f"Experiment result in '{output_dir}' has no runs"
        return cls(output_dir, data["experiment_name"], GitStamp.from_dict(data["git_stamp"]), data["runs"])

    @classmethod
    def from_run_outputs(
        cls,
        experiment_output_directory: str | Path,
        run_metadata_by_name: Mapping[str, Mapping[str, Any]],
        git_stamp: GitStamp | None = None,
        *,
        experiment_name: str,
    ) -> ArenaExperimentResult:
        """Collect declared Run outputs from an Experiment output directory.

        Args:
            experiment_output_directory: Experiment output root containing one directory per Run.
            run_metadata_by_name: Environment, policy variant, final status, start time, and end time for each declared Run.
            git_stamp: Arena git commit and branch; read from the environment when omitted.
            experiment_name: Name of the Experiment, a non-empty single-line string.

        Returns:
            The Experiment result holding each declared Run's metadata and episode results.
        """
        assert run_metadata_by_name, "run_metadata_by_name must not be empty"
        experiment_output_directory = Path(experiment_output_directory)
        runs = {
            run_name: cls._collect_run(experiment_output_directory, run_name, run_metadata)
            for run_name, run_metadata in run_metadata_by_name.items()
        }
        return cls(
            experiment_output_directory,
            experiment_name,
            git_stamp if git_stamp is not None else GitStamp.from_environment(),
            runs,
        )

    @property
    def all_runs_completed(self) -> bool:
        """Whether every Run has the completed status."""
        return all(run_data["status"] == RunStatus.COMPLETED.value for run_data in self.runs.values())

    @property
    def latest_run_end_timestamp_ns(self) -> int:
        """Latest Run end time in Unix epoch nanoseconds (UTC)."""
        return max(run_data["end_timestamp_ns"] for run_data in self.runs.values())

    @classmethod
    def _collect_run(
        cls, experiment_output_directory: Path, run_name: str, run_metadata: Mapping[str, Any]
    ) -> ArenaExperimentRunData:
        """Collect one declared Run's metadata and episode results."""
        cls.assert_run_name_is_safe_path_component(run_name)
        environment = run_metadata["environment"]
        status = run_metadata["status"]
        valid_statuses = [run_status.value for run_status in RunStatus]
        assert status in valid_statuses, f"status for Run {run_name!r} must be one of {valid_statuses}"

        start_timestamp_ns = run_metadata["start_timestamp_ns"]
        assert isinstance(start_timestamp_ns, int) and not isinstance(
            start_timestamp_ns, bool
        ), f"start_timestamp_ns for Run {run_name!r} must be an int"
        end_timestamp_ns = run_metadata["end_timestamp_ns"]
        assert isinstance(end_timestamp_ns, int) and not isinstance(
            end_timestamp_ns, bool
        ), f"end_timestamp_ns for Run {run_name!r} must be an int"

        run_output_directory = experiment_output_directory / run_name
        if status == RunStatus.COMPLETED.value:
            assert (
                run_output_directory.is_dir()
            ), f"completed Run {run_name!r} is missing its output directory: {run_output_directory}"
        elif run_output_directory.exists():
            assert (
                run_output_directory.is_dir()
            ), f"output path for failed Run {run_name!r} must be a directory: {run_output_directory}"

        rebuilds = (
            _collect_rebuilds(run_output_directory, experiment_output_directory)
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
            "start_timestamp_ns": start_timestamp_ns,
            "end_timestamp_ns": end_timestamp_ns,
            "rebuilds": rebuilds,
        }

    def to_dict(self) -> ArenaExperimentResultData:
        """Return the JSON-compatible Experiment result."""
        return {"git_stamp": self.git_stamp.to_dict(), "experiment_name": self.experiment_name, "runs": self.runs}

    def write(self) -> Path:
        """Write the Experiment result to ``arena_experiment_result.json`` in the Experiment output directory."""
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
