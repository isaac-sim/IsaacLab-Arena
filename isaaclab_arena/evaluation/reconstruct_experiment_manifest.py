# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct an Experiment manifest for results recorded before manifests were written.

The Experiment Runner now writes ``experiment_manifest.json`` from the composed configuration, which
is the authoritative source of task and policy labels. Older output directories have no such record,
so this module recovers what it can from the directory layout and the per-episode results: the task
and policy labels are factorized out of the Run names, and the remaining fields are read back from
the recorded episodes.

Depends only on the standard library so it can run on the host, without the Docker container.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from isaaclab_arena.evaluation.experiment_manifest import (
    ExperimentManifest,
    LabelSource,
    ManifestSource,
    RunManifestEntry,
    write_experiment_manifest,
)
from isaaclab_arena.recording.episode_results_files import (
    find_episode_results_files,
    parse_episode_results_rebuild_index,
    read_episode_results,
)

# Longest policy-name suffix considered when factorizing Run names, in underscore-separated tokens.
# Covers multi-token policy labels such as ``pi0_remote`` without letting a task name's trailing
# tokens be mistaken for a policy.
_MAX_POLICY_NAME_TOKENS = 3


def split_run_names_with_policy_names(
    run_names: list[str],
    policy_names: list[str],
) -> dict[str, tuple[str, str]] | None:
    """Split Run names into task and policy using a known set of policy names.

    Args:
        run_names: Run names to split.
        policy_names: Policy labels each Run name is expected to end with, after an underscore.

    Returns:
        Run name -> (task, policy), or ``None`` when some Run matches no policy name.
    """
    labels: dict[str, tuple[str, str]] = {}
    for run_name in run_names:
        matched = [policy for policy in policy_names if run_name.endswith(f"_{policy}")]
        if not matched:
            return None
        # Prefer the longest match so "pi0" cannot shadow a "remote_pi0" style label.
        policy = max(matched, key=len)
        labels[run_name] = (run_name[: -(len(policy) + 1)], policy)
    return labels


def infer_task_and_policy_labels(run_names: list[str]) -> dict[str, tuple[str, str]] | None:
    """Factorize Run names into task and policy labels by trying each trailing-token split.

    A split is accepted only when the Run names form a complete task x policy grid over at least two
    policies, which is strong evidence that the trailing tokens really are a policy axis rather than
    part of the task name. The shortest such suffix wins.

    Args:
        run_names: Run names to factorize.

    Returns:
        Run name -> (task, policy), or ``None`` when no split yields a complete grid.
    """
    for num_tokens in range(1, _MAX_POLICY_NAME_TOKENS + 1):
        candidate: dict[str, tuple[str, str]] = {}
        for run_name in run_names:
            tokens = run_name.split("_")
            if len(tokens) <= num_tokens:
                candidate = {}
                break
            candidate[run_name] = ("_".join(tokens[:-num_tokens]), "_".join(tokens[-num_tokens:]))
        if not candidate:
            continue
        tasks = {task for task, _ in candidate.values()}
        policies = {policy for _, policy in candidate.values()}
        if len(policies) >= 2 and len(run_names) == len(tasks) * len(policies):
            return candidate
    return None


def _summarize_recorded_runs(run_dir: Path) -> dict:
    """Return the fields recoverable from one Run directory's per-episode results."""
    records = []
    rebuild_indices = set()
    for results_path in find_episode_results_files(run_dir):
        rebuild_indices.add(parse_episode_results_rebuild_index(results_path.name))
        records.extend(read_episode_results(results_path))

    env_ids = {record["env_id"] for record in records if "env_id" in record}
    instructions = {record["language_instruction"] for record in records if record.get("language_instruction")}
    variation_values: dict[str, set] = {}
    for record in records:
        for key, value in (record.get("variations") or {}).items():
            variation_values.setdefault(key, set()).add(value)

    return {
        "num_episodes": len(records) or None,
        "num_rebuilds": len(rebuild_indices) or None,
        "num_envs": len(env_ids) or None,
        "language_instruction": instructions.pop() if len(instructions) == 1 else None,
        # A variation applied uniformly is recorded as its single value; one that was swept is
        # recorded as the sorted set of values that actually appeared.
        "variations": {
            key: next(iter(values)) if len(values) == 1 else sorted(values, key=str)
            for key, values in sorted(variation_values.items())
        },
    }


def find_run_directories(experiment_dir: Path) -> list[str]:
    """Return the names of ``experiment_dir`` sub-directories that hold recorded Run output.

    Args:
        experiment_dir: Experiment output directory to scan.
    """
    run_names = []
    for child in sorted(experiment_dir.iterdir()):
        if not child.is_dir():
            continue
        if find_episode_results_files(child) or next(child.glob("*.mp4"), None) is not None:
            run_names.append(child.name)
    return run_names


def reconstruct_experiment_manifest(
    experiment_dir: str | Path,
    *,
    policy_names: list[str] | None = None,
    experiment_name: str | None = None,
    created_at: str | None = None,
) -> ExperimentManifest:
    """Reconstruct a manifest for an Experiment output directory written before manifests existed.

    Args:
        experiment_dir: Experiment output directory holding one sub-directory per Run.
        policy_names: Known policy labels to split Run names on, inferred when omitted.
        experiment_name: Name recorded for the Experiment, defaulting to the directory name.
        created_at: ISO-8601 creation timestamp, defaulting to now.

    Returns:
        The reconstructed manifest, marked as such so consumers know its labels were not configured.
    """
    experiment_dir = Path(experiment_dir)
    assert experiment_dir.is_dir(), f"Experiment directory does not exist: '{experiment_dir}'"

    run_names = find_run_directories(experiment_dir)
    assert run_names, f"No Run sub-directories with recorded results or videos found in '{experiment_dir}'"

    if policy_names:
        labels = split_run_names_with_policy_names(run_names, policy_names)
        assert labels is not None, (
            f"Some Run names in '{experiment_dir}' do not end with any of the given policy names "
            f"{sorted(policy_names)}. Pass the correct --policy_names, or omit them to infer the split."
        )
    else:
        labels = infer_task_and_policy_labels(run_names)
        assert labels is not None, (
            f"Could not factorize the {len(run_names)} Run names in '{experiment_dir}' into a complete "
            "task x policy grid. Pass --policy_names to state the policy labels explicitly."
        )

    runs = []
    for run_name in run_names:
        task, policy = labels[run_name]
        runs.append(
            RunManifestEntry(
                name=run_name,
                task=task,
                policy=policy,
                **_summarize_recorded_runs(experiment_dir / run_name),
            )
        )

    return ExperimentManifest(
        runs=runs,
        experiment_name=experiment_name if experiment_name is not None else experiment_dir.name,
        created_at=created_at if created_at is not None else datetime.now().isoformat(timespec="seconds"),
        source=ManifestSource.RECONSTRUCTED,
        label_source=LabelSource.INFERRED_FROM_RUN_NAMES,
    )


def _print_manifest_summary(manifest: ExperimentManifest) -> None:
    """Print the reconstructed grouping so it can be eyeballed before the manifest is trusted."""
    total_episodes = sum(run.num_episodes or 0 for run in manifest.runs)
    print(
        f"Reconstructed {len(manifest.runs)} run(s): {len(manifest.tasks)} task(s) x "
        f"{len(manifest.policies)} policy(ies), {total_episodes} episode(s)."
    )
    print(f"Policies: {', '.join(manifest.policies)}")
    episodes_by_policy = Counter()
    for run in manifest.runs:
        episodes_by_policy[run.policy] += run.num_episodes or 0
    for policy, num_episodes in episodes_by_policy.items():
        print(f"  {policy}: {num_episodes} episode(s)")
    preview = manifest.tasks[:5]
    suffix = ", ..." if len(manifest.tasks) > len(preview) else ""
    print(f"Tasks: {', '.join(preview)}{suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an experiment_manifest.json for an Experiment output directory recorded before"
            " the Experiment Runner wrote manifests. Task and policy labels are factorized out of the Run"
            " directory names and the remaining fields are read back from the recorded episodes."
        )
    )
    parser.add_argument(
        "--experiment_dir",
        required=True,
        type=Path,
        help="Experiment output directory holding one sub-directory per Run.",
    )
    parser.add_argument(
        "--policy_names",
        nargs="+",
        default=None,
        help=(
            "Policy labels that Run names end with (e.g. --policy_names pi0 cosmos)."
            " Inferred from the Run names when omitted."
        ),
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Name recorded for the Experiment. Defaults to the directory name.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the reconstructed grouping without writing the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = reconstruct_experiment_manifest(
        args.experiment_dir,
        policy_names=args.policy_names,
        experiment_name=args.experiment_name,
    )
    _print_manifest_summary(manifest)
    if args.dry_run:
        print("Dry run: no manifest written.")
        return
    manifest_path = write_experiment_manifest(manifest, args.experiment_dir)
    print(f"Wrote reconstructed Experiment manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
