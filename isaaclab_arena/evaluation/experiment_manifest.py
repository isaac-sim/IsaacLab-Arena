# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Describe the Runs of an Experiment in a manifest written beside its recorded results.

Reading and writing manifests depends only on the standard library, so reporting and analysis tools
can consume them without a running SimulationApp. Only ``build_experiment_manifest`` needs one,
because deriving labels resolves the environment and policy registries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg

EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
"""Name of the manifest file written into an Experiment's output directory."""

MANIFEST_SCHEMA_VERSION = 1
"""Schema version of the manifest document, incremented on incompatible changes."""


class ManifestSource(str, Enum):
    """Record how a manifest was produced, because it bounds how much it can be trusted."""

    EXPERIMENT_RUNNER = "experiment_runner"
    """Written from the composed Experiment configuration before its Runs executed."""

    RECONSTRUCTED = "reconstructed"
    """Rebuilt after the fact from an output directory whose configuration was not recorded."""


class LabelSource(str, Enum):
    """Record where a manifest's task and policy labels came from."""

    CONFIGURED = "configured"
    """Taken from explicit ``RunLabelsCfg`` values in the Experiment configuration."""

    DERIVED = "derived"
    """Derived from each Run's environment and policy configuration."""

    INFERRED_FROM_RUN_NAMES = "inferred_from_run_names"
    """Factorized out of the Run names, because no configuration was available."""


@dataclass
class RunManifestEntry:
    """Identify one Run and the grouping labels used to aggregate it with others."""

    name: str
    """Run name, which is also its output sub-directory name and its ``job_name`` in results."""

    task: str
    """Label grouping this Run with other Runs evaluating the same task."""

    policy: str
    """Label grouping this Run with other Runs evaluating the same policy."""

    environment: str | None = None
    """Environment graph-spec path or registered environment name, when known."""

    policy_type: str | None = None
    """Registered policy name or dotted class path, when known."""

    num_episodes: int | None = None
    """Configured episode budget, or the recorded episode count for a reconstructed manifest."""

    num_rebuilds: int | None = None
    """Number of fresh environment constructions the episode budget was split across."""

    num_envs: int | None = None
    """Number of parallel environments the Run was rolled out in, or the number observed in a
    reconstructed manifest, which understates it when an environment recorded no episode."""

    language_instruction: str | None = None
    """Instruction given to the policy, when the Run or its records carry one."""

    variations: dict[str, Any] = field(default_factory=dict)
    """Variation values applied to the environment. A reconstructed manifest records the values that
    actually appeared in the episodes rather than the configured variation spec."""


@dataclass
class ExperimentManifest:
    """Describe every Run of one Experiment so its results can be grouped without re-reading configs."""

    runs: list[RunManifestEntry]
    """One entry per Run, in Experiment declaration order."""

    experiment_name: str = ""
    """Name of the Experiment, defaulting to its output directory name."""

    created_at: str = ""
    """ISO-8601 timestamp of when the manifest was written."""

    source: ManifestSource = ManifestSource.EXPERIMENT_RUNNER
    """How the manifest was produced."""

    label_source: LabelSource = LabelSource.DERIVED
    """Where the task and policy labels came from."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    """Schema version of this document."""

    @property
    def tasks(self) -> list[str]:
        """Return the distinct task labels, in first-seen order."""
        return list(dict.fromkeys(run.task for run in self.runs))

    @property
    def policies(self) -> list[str]:
        """Return the distinct policy labels, in first-seen order."""
        return list(dict.fromkeys(run.policy for run in self.runs))

    def run_by_name(self, run_name: str) -> RunManifestEntry | None:
        """Return the entry for ``run_name``, or ``None`` when the Run is not described.

        Args:
            run_name: Name of the Run to look up.
        """
        for run in self.runs:
            if run.name == run_name:
                return run
        return None


def derive_task_label(run_cfg: ArenaRunCfg) -> str:
    """Return the task label for ``run_cfg``, preferring its configured label over a derived one.

    Graph-spec environments derive their label from the task YAML stem (``banana_in_bowl.yaml`` ->
    ``banana_in_bowl``). Registered environments derive theirs from the factory name, which is shared
    by every Run of that environment family; set ``labels.task`` explicitly to distinguish them.

    Args:
        run_cfg: Run whose task label is needed.
    """
    if run_cfg.labels.task:
        return run_cfg.labels.task

    # Imported here because the registries import environment modules that require a running
    # SimulationApp, while reading and reconstructing manifests must work without one.
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg

    if isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg):
        return Path(run_cfg.environment.env_graph_spec_yaml_path).stem
    return EnvironmentRegistry().get_factory_type_for_cfg(run_cfg.environment).name


def derive_policy_label(run_cfg: ArenaRunCfg) -> str:
    """Return the policy label for ``run_cfg``, preferring its configured label over a derived one.

    The derived label is the registered policy name (e.g. ``pi0_remote``), which does not capture the
    served checkpoint for remote policies; set ``labels.policy`` explicitly to compare checkpoints.

    Args:
        run_cfg: Run whose policy label is needed.
    """
    if run_cfg.labels.policy:
        return run_cfg.labels.policy

    from isaaclab_arena.assets.registries import PolicyRegistry

    return PolicyRegistry().get_policy_type_for_cfg(run_cfg.policy).name


def _environment_identifier(run_cfg: ArenaRunCfg) -> str:
    """Return the graph-spec path or registered name identifying a Run's environment."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg

    if isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg):
        return run_cfg.environment.env_graph_spec_yaml_path
    return EnvironmentRegistry().get_factory_type_for_cfg(run_cfg.environment).name


def _policy_type_identifier(run_cfg: ArenaRunCfg) -> str:
    """Return the registered name, or dotted class path for out-of-tree policies."""
    from isaaclab_arena.assets.registries import PolicyRegistry

    policy_type = PolicyRegistry().get_policy_type_for_cfg(run_cfg.policy)
    if policy_type.__module__.startswith("isaaclab_arena.policy."):
        return policy_type.name
    return f"{policy_type.__module__}.{policy_type.__qualname__}"


def build_experiment_manifest(
    experiment_cfg: ArenaExperimentCfg,
    *,
    experiment_name: str = "",
    created_at: str | None = None,
) -> ExperimentManifest:
    """Build a manifest describing every Run of a composed Experiment.

    Must be called with a running SimulationApp, because deriving labels resolves the environment and
    policy registries.

    Args:
        experiment_cfg: Composed Experiment whose Runs are described.
        experiment_name: Name recorded for the Experiment, usually its output directory name.
        created_at: ISO-8601 creation timestamp, defaulting to now.

    Returns:
        The manifest, with one entry per Run in declaration order.
    """
    runs = [
        RunManifestEntry(
            name=run_cfg.name,
            task=derive_task_label(run_cfg),
            policy=derive_policy_label(run_cfg),
            environment=_environment_identifier(run_cfg),
            policy_type=_policy_type_identifier(run_cfg),
            num_episodes=run_cfg.rollout_limit.num_episodes,
            num_rebuilds=run_cfg.num_rebuilds,
            num_envs=run_cfg.environment_builder.num_envs,
            language_instruction=run_cfg.environment_builder.language_instruction,
            variations=dict(run_cfg.variations),
        )
        for run_cfg in experiment_cfg.runs.values()
    ]
    any_configured_label = any(run_cfg.labels.task or run_cfg.labels.policy for run_cfg in experiment_cfg.runs.values())
    return ExperimentManifest(
        runs=runs,
        experiment_name=experiment_name,
        created_at=created_at if created_at is not None else datetime.now().isoformat(timespec="seconds"),
        source=ManifestSource.EXPERIMENT_RUNNER,
        label_source=LabelSource.CONFIGURED if any_configured_label else LabelSource.DERIVED,
    )


def write_experiment_manifest(manifest: ExperimentManifest, output_dir: str | Path) -> Path:
    """Write ``manifest`` into ``output_dir`` and return the path written.

    Args:
        manifest: Manifest to serialize.
        output_dir: Experiment output directory the manifest belongs to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / EXPERIMENT_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return manifest_path


def read_experiment_manifest(directory: str | Path) -> ExperimentManifest | None:
    """Read the manifest in ``directory``, returning ``None`` when the Experiment predates manifests.

    Args:
        directory: Experiment output directory to read the manifest from.
    """
    manifest_path = Path(directory) / EXPERIMENT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document.get("schema_version") == MANIFEST_SCHEMA_VERSION, (
        f"Unsupported Experiment manifest schema version {document.get('schema_version')!r} in "
        f"'{manifest_path}'; this build reads version {MANIFEST_SCHEMA_VERSION}."
    )
    entry_field_names = set(RunManifestEntry.__dataclass_fields__)
    runs = [
        RunManifestEntry(**{key: value for key, value in run.items() if key in entry_field_names})
        for run in document["runs"]
    ]
    return ExperimentManifest(
        runs=runs,
        experiment_name=document.get("experiment_name", ""),
        created_at=document.get("created_at", ""),
        source=ManifestSource(document.get("source", ManifestSource.EXPERIMENT_RUNNER.value)),
        label_source=LabelSource(document.get("label_source", LabelSource.DERIVED.value)),
        schema_version=document["schema_version"],
    )
