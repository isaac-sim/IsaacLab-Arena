# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, RolloutLimitCfg, RunLabelsCfg
from isaaclab_arena.evaluation.experiment_manifest import (
    EXPERIMENT_MANIFEST_FILENAME,
    ExperimentManifest,
    LabelSource,
    ManifestSource,
    RunManifestEntry,
    build_experiment_manifest,
    read_experiment_manifest,
    write_experiment_manifest,
)
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.evaluation.reconstruct_experiment_manifest import (
    find_run_directories,
    infer_task_and_policy_labels,
    reconstruct_experiment_manifest,
    split_run_names_with_policy_names,
)
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg


def _write_run_results(
    experiment_dir,
    run_name: str,
    *,
    num_episodes: int = 4,
    num_envs: int = 2,
    instruction: str = "pick up the banana",
    hdr: str = "home_office_robolab",
):
    """Write one Run sub-directory holding a single rebuild's per-episode results."""
    run_dir = experiment_dir / run_name
    run_dir.mkdir(parents=True)
    lines = []
    for episode_index in range(num_episodes):
        lines.append(
            json.dumps({
                "job_name": run_name,
                "env_id": episode_index % num_envs,
                "episode_in_env": episode_index // num_envs,
                "success": episode_index % 2 == 0,
                "language_instruction": instruction,
                "variations": {"light.hdr_image": hdr},
            })
        )
    (run_dir / "episode_results_rebuild0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_manifest_roundtrips_through_json(tmp_path):
    manifest = ExperimentManifest(
        runs=[
            RunManifestEntry(
                name="banana_in_bowl_pi0",
                task="banana_in_bowl",
                policy="pi0",
                environment="isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml",
                policy_type="isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy",
                num_episodes=300,
                num_rebuilds=1,
                num_envs=25,
                language_instruction="pick up the banana and place it in the bowl",
                variations={"light": {"hdr_image": {"hdr_names": ["home_office_robolab"]}}},
            )
        ],
        experiment_name="two_policies",
        created_at="2026-08-06T12:00:00",
        source=ManifestSource.EXPERIMENT_RUNNER,
        label_source=LabelSource.DERIVED,
    )

    manifest_path = write_experiment_manifest(manifest, tmp_path)
    assert manifest_path == tmp_path / EXPERIMENT_MANIFEST_FILENAME

    loaded = read_experiment_manifest(tmp_path)
    assert loaded == manifest
    assert loaded.tasks == ["banana_in_bowl"]
    assert loaded.policies == ["pi0"]
    assert loaded.run_by_name("banana_in_bowl_pi0").num_episodes == 300
    assert loaded.run_by_name("absent") is None


def test_read_manifest_returns_none_when_experiment_predates_manifests(tmp_path):
    assert read_experiment_manifest(tmp_path) is None


def test_read_manifest_rejects_unsupported_schema_version(tmp_path):
    (tmp_path / EXPERIMENT_MANIFEST_FILENAME).write_text(
        json.dumps({"schema_version": 99, "runs": []}), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="Unsupported Experiment manifest schema version"):
        read_experiment_manifest(tmp_path)


def test_infer_labels_factorizes_a_complete_task_policy_grid():
    labels = infer_task_and_policy_labels([
        "banana_in_bowl_pi0",
        "banana_in_bowl_cosmos",
        "bagels_on_plate_pi0",
        "bagels_on_plate_cosmos",
    ])

    assert labels == {
        "banana_in_bowl_pi0": ("banana_in_bowl", "pi0"),
        "banana_in_bowl_cosmos": ("banana_in_bowl", "cosmos"),
        "bagels_on_plate_pi0": ("bagels_on_plate", "pi0"),
        "bagels_on_plate_cosmos": ("bagels_on_plate", "cosmos"),
    }


def test_infer_labels_recovers_multi_token_policy_names():
    # A one-token split leaves a single policy ("remote"), so only the two-token split is a grid.
    labels = infer_task_and_policy_labels([
        "banana_in_bowl_pi0_remote",
        "banana_in_bowl_cosmos_remote",
        "bagels_on_plate_pi0_remote",
        "bagels_on_plate_cosmos_remote",
    ])

    assert labels["banana_in_bowl_pi0_remote"] == ("banana_in_bowl", "pi0_remote")
    assert labels["bagels_on_plate_cosmos_remote"] == ("bagels_on_plate", "cosmos_remote")


def test_infer_labels_rejects_run_names_that_are_not_a_grid():
    # Three tasks against one policy is not evidence of a policy axis.
    assert infer_task_and_policy_labels(["banana_in_bowl_pi0", "bagels_on_plate_pi0", "bowl_in_bin_pi0"]) is None
    # An incomplete grid must not be silently accepted.
    assert infer_task_and_policy_labels(["banana_in_bowl_pi0", "banana_in_bowl_cosmos", "bagels_on_plate_pi0"]) is None


def test_split_run_names_with_explicit_policy_names_prefers_the_longest_match():
    labels = split_run_names_with_policy_names(
        ["banana_in_bowl_pi0", "banana_in_bowl_remote_pi0"],
        ["pi0", "remote_pi0"],
    )

    assert labels == {
        "banana_in_bowl_pi0": ("banana_in_bowl", "pi0"),
        "banana_in_bowl_remote_pi0": ("banana_in_bowl", "remote_pi0"),
    }


def test_split_run_names_returns_none_when_a_run_matches_no_policy():
    assert split_run_names_with_policy_names(["banana_in_bowl_gr00t"], ["pi0", "cosmos"]) is None


def test_reconstruct_manifest_recovers_grouping_and_recorded_fields(tmp_path):
    for task in ("banana_in_bowl", "bagels_on_plate"):
        for policy in ("pi0", "cosmos"):
            _write_run_results(tmp_path, f"{task}_{policy}", num_episodes=4, num_envs=2)
    # Files that are not Run output must not be mistaken for Runs.
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "empty_dir").mkdir()

    manifest = reconstruct_experiment_manifest(tmp_path, created_at="2026-08-06T12:00:00")

    assert manifest.source is ManifestSource.RECONSTRUCTED
    assert manifest.label_source is LabelSource.INFERRED_FROM_RUN_NAMES
    assert manifest.experiment_name == tmp_path.name
    assert sorted(manifest.tasks) == ["bagels_on_plate", "banana_in_bowl"]
    assert sorted(manifest.policies) == ["cosmos", "pi0"]
    assert len(manifest.runs) == 4

    run = manifest.run_by_name("banana_in_bowl_pi0")
    assert (run.task, run.policy) == ("banana_in_bowl", "pi0")
    assert (run.num_episodes, run.num_envs, run.num_rebuilds) == (4, 2, 1)
    assert run.language_instruction == "pick up the banana"
    # A variation applied uniformly is recorded as its single value.
    assert run.variations == {"light.hdr_image": "home_office_robolab"}
    # Nothing about the configuration survives in an old output directory.
    assert run.environment is None and run.policy_type is None


def test_reconstruct_manifest_records_a_swept_variation_as_its_distinct_values(tmp_path):
    for policy in ("pi0", "cosmos"):
        run_dir = tmp_path / f"banana_in_bowl_{policy}"
        run_dir.mkdir()
        lines = [
            json.dumps({"env_id": 0, "episode_in_env": index, "variations": {"light.hdr_image": hdr}})
            for index, hdr in enumerate(("home_office_robolab", "warehouse"))
        ]
        (run_dir / "episode_results_rebuild0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # A second task keeps the Run names a complete grid.
        _write_run_results(tmp_path, f"bagels_on_plate_{policy}")

    manifest = reconstruct_experiment_manifest(tmp_path)

    assert manifest.run_by_name("banana_in_bowl_pi0").variations == {
        "light.hdr_image": ["home_office_robolab", "warehouse"]
    }


def test_reconstruct_manifest_counts_rebuilds_and_is_readable_after_writing(tmp_path):
    for policy in ("pi0", "cosmos"):
        run_dir = _write_run_results(tmp_path, f"banana_in_bowl_{policy}")
        (run_dir / "episode_results_rebuild1.jsonl").write_text(
            json.dumps({"env_id": 0, "episode_in_env": 0, "success": True}) + "\n", encoding="utf-8"
        )
        _write_run_results(tmp_path, f"bagels_on_plate_{policy}")

    manifest = reconstruct_experiment_manifest(tmp_path)
    write_experiment_manifest(manifest, tmp_path)

    assert manifest.run_by_name("banana_in_bowl_pi0").num_rebuilds == 2
    assert manifest.run_by_name("banana_in_bowl_pi0").num_episodes == 5
    assert read_experiment_manifest(tmp_path) == manifest


def test_find_run_directories_includes_video_only_runs(tmp_path):
    (tmp_path / "banana_in_bowl_pi0").mkdir()
    (tmp_path / "banana_in_bowl_pi0" / "robot-cam-rebuild0-env0-wrist_cam-episode-0.mp4").write_bytes(b"")
    _write_run_results(tmp_path, "bagels_on_plate_pi0")
    (tmp_path / "not_a_run").mkdir()

    assert find_run_directories(tmp_path) == ["bagels_on_plate_pi0", "banana_in_bowl_pi0"]


def test_reconstruct_manifest_rejects_run_names_it_cannot_factorize(tmp_path):
    _write_run_results(tmp_path, "banana_in_bowl_pi0")
    _write_run_results(tmp_path, "bagels_on_plate_pi0")

    with pytest.raises(AssertionError, match="Could not factorize"):
        reconstruct_experiment_manifest(tmp_path)


def test_reconstruct_manifest_rejects_a_directory_with_no_runs(tmp_path):
    with pytest.raises(AssertionError, match="No Run sub-directories"):
        reconstruct_experiment_manifest(tmp_path)


def test_build_manifest_derives_labels_from_a_graph_spec_environment_and_policy():
    run_cfg = ArenaRunCfg(
        name="banana_in_bowl_zero",
        environment=LegacyGraphEnvironmentCfg(
            env_graph_spec_yaml_path="isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml",
            per_run_overrides={"enable_cameras": True},
        ),
        policy=ZeroActionPolicyCfg(),
        rollout_limit=RolloutLimitCfg(num_episodes=300),
    )

    manifest = build_experiment_manifest(
        ArenaExperimentCfg(runs={run_cfg.name: run_cfg}),
        experiment_name="robolab",
        created_at="2026-08-06T12:00:00",
    )

    assert manifest.source is ManifestSource.EXPERIMENT_RUNNER
    assert manifest.label_source is LabelSource.DERIVED
    run = manifest.run_by_name("banana_in_bowl_zero")
    # The task label comes from the graph-spec YAML stem, not from the Run name.
    assert (run.task, run.policy) == ("banana_in_bowl", "zero_action")
    assert run.environment == "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
    assert run.policy_type == "zero_action"
    assert run.num_episodes == 300


def test_build_manifest_derives_the_task_label_from_a_registered_environment():
    run_cfg = ArenaRunCfg(
        name="baseline",
        environment=PickAndPlaceMapleTableEnvironmentCfg(),
        policy=ZeroActionPolicyCfg(),
    )

    manifest = build_experiment_manifest(ArenaExperimentCfg(runs={run_cfg.name: run_cfg}))

    run = manifest.run_by_name("baseline")
    assert (run.task, run.policy) == ("pick_and_place_maple_table", "zero_action")


def test_build_manifest_prefers_configured_labels_over_derived_ones():
    run_cfg = ArenaRunCfg(
        name="baseline",
        environment=PickAndPlaceMapleTableEnvironmentCfg(),
        policy=ZeroActionPolicyCfg(),
        labels=RunLabelsCfg(task="banana_in_bowl", policy="pi05_checkpoint_42"),
    )

    manifest = build_experiment_manifest(ArenaExperimentCfg(runs={run_cfg.name: run_cfg}))

    assert manifest.label_source is LabelSource.CONFIGURED
    run = manifest.run_by_name("baseline")
    assert (run.task, run.policy) == ("banana_in_bowl", "pi05_checkpoint_42")
    # The derived identity is still recorded, so an explicit label never hides what actually ran.
    assert run.environment == "pick_and_place_maple_table"
    assert run.policy_type == "zero_action"


def test_configured_labels_survive_the_experiment_yaml_roundtrip(tmp_path):
    # OSMO serializes the composed Experiment to YAML and reloads it in the container, so labels that
    # did not survive that trip would be silently dropped for exactly the large Experiments they matter for.
    run_cfg = ArenaRunCfg(
        name="baseline",
        environment=PickAndPlaceMapleTableEnvironmentCfg(),
        policy=ZeroActionPolicyCfg(),
        labels=RunLabelsCfg(task="banana_in_bowl", policy="pi05_ckpt42"),
    )
    experiment_yaml = tmp_path / "experiment.yaml"
    experiment_yaml.write_text(
        serialize_arena_experiment_to_yaml(ArenaExperimentCfg(runs={run_cfg.name: run_cfg})), encoding="utf-8"
    )

    reloaded = load_arena_experiment_from_config_file(experiment_yaml, device="cpu")

    assert reloaded.runs["baseline"].labels == RunLabelsCfg(task="banana_in_bowl", policy="pi05_ckpt42")
