# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how recorded results are aggregated into the evaluation report's data model."""

import json

from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.video.camera_observation_video_recorder import format_episode_video_filename
from isaaclab_arena.visualization.report_data import (
    EpisodeSummary,
    JobSummary,
    RunExecutionReport,
    build_experiment_summary,
    infer_task_and_policy_labels,
)


def _progress(objectives: dict[str, int], events: list[tuple[str, int, str]], score: float) -> dict:
    """Build a ``progress`` block from objective group totals and (objective, index, name) events."""
    return {
        "overall_score": score,
        "objectives": {name: {"total_groups": total} for name, total in objectives.items()},
        "events": [
            {"objective": objective, "predicate_index": index, "predicate_name": name}
            for objective, index, name in events
        ],
    }


def _write_run(experiment_dir, run_name: str, records: list[dict], cameras: tuple[str, ...] = ("wrist_cam",)):
    """Write one Run sub-directory holding per-episode results and a video per (episode, camera)."""
    run_dir = experiment_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    for record in records:
        for camera in cameras:
            name = format_episode_video_filename(
                "robot-cam-rebuild0", record["env_id"], camera, record["episode_in_env"]
            )
            (run_dir / name).write_bytes(b"")
    return run_dir


def test_progress_fraction_normalizes_by_the_achievable_score():
    # A multi-object task declares one objective per object, so its achievable score is well above 1.
    episode = EpisodeSummary(
        env_index=0,
        episode_index=0,
        video_by_camera={},
        record={"progress": _progress({"a": 1, "b": 1, "c": 1}, [], score=1.5)},
    )

    assert episode.max_score == 3.0
    assert episode.progress_fraction == 0.5


def test_progress_fraction_is_none_without_recorded_objectives():
    episode = EpisodeSummary(env_index=0, episode_index=0, video_by_camera={}, record={"success": True})

    assert episode.max_score is None
    assert episode.progress_fraction is None


def test_funnel_counts_objective_instances_rather_than_events():
    # The same predicate fires once per object in a multi-object task, and twice for one of them.
    # Counting events would report 3 for a stage that only two objective instances reached.
    episode = EpisodeSummary(
        env_index=0,
        episode_index=0,
        video_by_camera={},
        record={
            "success": False,
            "progress": _progress(
                {"subtask_0": 1, "subtask_1": 1},
                [
                    ("subtask_0", 0, "objects_settled"),
                    ("subtask_0", 0, "objects_settled"),
                    ("subtask_1", 0, "objects_settled"),
                    ("subtask_0", 1, "object_is_above_height(object_name='lemon')"),
                ],
                score=0.5,
            ),
        },
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    stages = job.funnel
    assert [(stage.index, stage.name, stage.num_reached) for stage in stages] == [
        (0, "objects_settled", 2),
        (1, "object_is_above_height", 1),
    ]
    # Two objectives in one episode is two instances, which is what the stages are a fraction of.
    assert job.num_objective_instances == 2


def test_objectives_list_predicates_the_episode_never_reached():
    # The stalled episode's own events name only the first predicate. The rest of the sequence is
    # recovered from the episode that got further, so both cards can show the full sequence.
    complete = EpisodeSummary(
        0,
        0,
        {},
        {
            "success": True,
            "progress": _progress(
                {"pick_and_place": 1},
                [
                    ("pick_and_place", 0, "objects_settled"),
                    ("pick_and_place", 1, "object_is_above_height(object_name='banana')"),
                    ("pick_and_place", 2, "object_on_destination(force_threshold=0.1)"),
                ],
                score=1.0,
            ),
        },
    )
    stalled = EpisodeSummary(
        0,
        1,
        {},
        {
            "success": False,
            "progress": {
                "overall_score": 0.33,
                "objectives": {
                    "pick_and_place": {
                        "score": 0.33,
                        "is_complete": False,
                        "total_groups": 1,
                        "active_predicates": {"default_group": "object_is_above_height(object_name='banana')"},
                    }
                },
                "events": [{
                    "objective": "pick_and_place",
                    "predicate_index": 0,
                    "predicate_name": "objects_settled",
                    "step": 7,
                }],
            },
        },
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[complete, stalled])

    assert job.predicate_sequence == {
        0: "objects_settled",
        1: "object_is_above_height",
        2: "object_on_destination",
    }

    objective = job.objectives_for(stalled)[0]
    assert objective.num_triggered == 1
    assert [(s.name, s.triggered, s.blocked) for s in objective.signals] == [
        ("objects_settled", True, False),
        # The objective was still waiting on this one, which is where it stalled.
        ("object_is_above_height", False, True),
        ("object_on_destination", False, False),
    ]
    assert objective.signals[0].step == 7

    # A completed objective has every signal triggered and nothing blocked.
    done = job.objectives_for(complete)[0]
    assert done.num_triggered == 3
    assert not any(signal.blocked for signal in done.signals)
    # The full predicate text is kept, because it names the object on multi-object tasks.
    assert done.signals[1].detail == "object_is_above_height(object_name='banana')"


def test_objectives_are_listed_per_subtask_for_a_multi_object_task():
    episode = EpisodeSummary(
        0,
        0,
        {},
        {
            "success": False,
            "progress": {
                "overall_score": 1.0,
                "objectives": {
                    "subtask_0/pick_and_place": {"score": 1.0, "is_complete": True, "total_groups": 1},
                    "subtask_1/pick_and_place": {
                        "score": 0.33,
                        "is_complete": False,
                        "total_groups": 1,
                        "active_predicates": {"default_group": "object_is_above_height(object_name='lemon_2')"},
                    },
                },
                "events": [
                    {
                        "objective": "subtask_0/pick_and_place",
                        "predicate_index": 0,
                        "predicate_name": "objects_settled",
                    },
                    {
                        "objective": "subtask_0/pick_and_place",
                        "predicate_index": 1,
                        "predicate_name": "object_is_above_height(object_name='lime')",
                    },
                    {
                        "objective": "subtask_1/pick_and_place",
                        "predicate_index": 0,
                        "predicate_name": "objects_settled",
                    },
                ],
            },
        },
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    objectives = job.objectives_for(episode)
    assert [objective.name for objective in objectives] == [
        "subtask_0/pick_and_place",
        "subtask_1/pick_and_place",
    ]
    assert objectives[0].num_triggered == 2 and objectives[0].is_complete
    assert objectives[1].num_triggered == 1 and not objectives[1].is_complete
    # Only the stalled subtask is marked as blocked, on the predicate naming its own object.
    assert [signal.blocked for signal in objectives[0].signals] == [False, False]
    assert [signal.blocked for signal in objectives[1].signals] == [False, True]


def test_outcome_disagreeing_with_progress_is_detected():
    # The success term and the progress objectives are separate mechanisms and really do disagree in
    # recorded data, so the report has to be able to say so rather than look like it miscounted.
    complete_but_failed = EpisodeSummary(
        0, 0, {}, {"success": False, "progress": {"all_complete": True, "overall_score": 2.0}}
    )
    incomplete_but_passed = EpisodeSummary(
        0, 1, {}, {"success": True, "progress": {"all_complete": False, "overall_score": 0.5}}
    )
    agreeing = EpisodeSummary(0, 2, {}, {"success": True, "progress": {"all_complete": True}})
    no_progress_block = EpisodeSummary(0, 3, {}, {"success": True})

    assert complete_but_failed.outcome_disagrees_with_progress
    assert incomplete_but_passed.outcome_disagrees_with_progress
    assert not agreeing.outcome_disagrees_with_progress
    assert no_progress_block.all_objectives_complete is None
    assert not no_progress_block.outcome_disagrees_with_progress


def test_success_rate_ignores_episodes_that_carry_no_success_term():
    episodes = [
        EpisodeSummary(0, 0, {}, {"success": True}),
        EpisodeSummary(0, 1, {}, {"success": False}),
        EpisodeSummary(0, 2, {}, {}),
    ]
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=episodes)

    assert job.num_episodes == 3
    assert job.num_scored_episodes == 2
    assert job.success_rate == 0.5


def test_success_rate_is_none_when_nothing_was_scored():
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[EpisodeSummary(0, 0, {}, {})])

    assert job.success_rate is None


def test_summary_groups_runs_by_task_and_policy_factorized_from_run_names(tmp_path):
    for task in ("banana_in_bowl", "bowl_in_bin"):
        for policy in ("pi0", "cosmos"):
            _write_run(
                tmp_path,
                f"{task}_{policy}",
                [{"env_id": 0, "episode_in_env": index, "success": index == 0} for index in range(2)],
            )

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.grouping_source == "run_names"
    assert [task.name for task in summary.tasks] == ["banana_in_bowl", "bowl_in_bin"]
    assert summary.policies == ["cosmos", "pi0"]
    assert summary.overall_success_rate == 0.5
    assert summary.success_rate_for_policy("pi0") == 0.5
    assert summary.num_episodes_for_policy("pi0") == 4
    assert summary.tasks[0].job_for_policy("pi0").name == "banana_in_bowl_pi0"
    assert summary.tasks[0].job_for_policy("absent") is None


def test_infer_labels_rejects_run_names_that_are_not_a_grid():
    # Three tasks against one policy is not evidence of a policy axis.
    assert infer_task_and_policy_labels(["banana_in_bowl_pi0", "bagels_on_plate_pi0", "bowl_in_bin_pi0"]) is None
    # An incomplete grid must not be silently accepted.
    assert infer_task_and_policy_labels(["banana_in_bowl_pi0", "banana_in_bowl_cosmos", "bagels_on_plate_pi0"]) is None


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


def test_summary_leaves_runs_ungrouped_when_no_labels_can_be_established(tmp_path):
    _write_run(tmp_path, "solo_run", [{"env_id": 0, "episode_in_env": 0, "success": True}])

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.grouping_source == "none"
    assert summary.is_grouped is False
    assert [task.name for task in summary.tasks] == ["solo_run"]


def test_summary_excludes_runs_whose_process_failed(tmp_path):
    _write_run(tmp_path, "good_pi0", [{"env_id": 0, "episode_in_env": 0, "success": True}])
    _write_run(tmp_path, "broken_pi0", [{"env_id": 0, "episode_in_env": 0, "success": False}])

    summary = build_experiment_summary(
        tmp_path,
        "Report",
        [RunExecutionReport(run_name="broken_pi0", status=RunStatus.FAILED, process_exit_code=17)],
    )

    assert [job.name for job in summary.jobs] == ["good_pi0"]
    assert summary.num_episodes == 1


def test_summary_pairs_videos_with_their_episode_records(tmp_path):
    _write_run(
        tmp_path,
        "banana_in_bowl_pi0",
        [{"env_id": 0, "episode_in_env": 0, "success": True}],
        cameras=("wrist_cam", "front_cam"),
    )
    _write_run(tmp_path, "banana_in_bowl_cosmos", [{"env_id": 0, "episode_in_env": 0, "success": False}])

    summary = build_experiment_summary(tmp_path, "Report")

    job = summary.tasks[0].job_for_policy("pi0")
    assert job.cameras == ["front_cam", "wrist_cam"]
    assert job.num_videos == 2
    episode = job.episodes[0]
    assert episode.success is True
    assert episode.video_by_camera["wrist_cam"].startswith("banana_in_bowl_pi0/")


def test_summary_renumbers_episodes_across_rebuilds(tmp_path):
    run_dir = tmp_path / "banana_in_bowl_pi0"
    run_dir.mkdir()
    for rebuild in (0, 1):
        (run_dir / f"episode_results_rebuild{rebuild}.jsonl").write_text(
            json.dumps({"env_id": 0, "episode_in_env": 0, "success": rebuild == 0}) + "\n", encoding="utf-8"
        )
    _write_run(tmp_path, "banana_in_bowl_cosmos", [{"env_id": 0, "episode_in_env": 0, "success": False}])

    summary = build_experiment_summary(tmp_path, "Report")

    job = summary.tasks[0].job_for_policy("pi0")
    # Both rebuilds recorded "episode 0"; they become episodes 0 and 1 of the same environment.
    assert [episode.episode_index for episode in job.episodes] == [0, 1]
    assert [episode.success for episode in job.episodes] == [True, False]
