# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test how recorded results are aggregated into the evaluation report data model."""

import json

import pytest

from isaaclab_arena.visualization.episode_results_files import format_episode_video_filename
from isaaclab_arena.visualization.report_data import (
    EpisodeIdentity,
    EpisodeSummary,
    JobSummary,
    RunExecutionReport,
    _infer_labels_from_explicit_suffixes,
    _infer_task_and_policy_labels_with_source,
    build_experiment_summary,
    normalize_run_status,
)


def _identity(env: int = 0, episode: int = 0, rebuild: int = 0, source: str = "") -> EpisodeIdentity:
    return EpisodeIdentity(source, rebuild, env, episode)


def _episode(record: dict | None = None, env: int = 0, episode: int = 0) -> EpisodeSummary:
    return EpisodeSummary(_identity(env, episode), episode, {}, record or {})


def _event(criteria_name: str, sequence_name: str, index: int, predicate_name: str, **fields) -> dict:
    """Build one recorded predicate event."""
    return {
        "criteria_name": criteria_name,
        "sequence_name": sequence_name,
        "predicate_index": index,
        "predicate_name": predicate_name,
        **fields,
    }


def _progress(
    active_predicate_by_criteria: dict[str, str | None], events: list[tuple[str, int, str]], score: float
) -> dict:
    """Build a one-sequence ``progress`` block from active predicates and (criteria_name, index, name) events.

    An active predicate of None marks the criteria as complete.
    """
    return {
        "overall_score": score,
        "criteria_by_name": {
            name: {
                "score": score,
                "is_complete": active_predicate is None,
                "total_sequences": 1,
                "active_predicates": {"default_sequence": active_predicate},
            }
            for name, active_predicate in active_predicate_by_criteria.items()
        },
        "events": [_event(criteria_name, "default_sequence", index, name) for criteria_name, index, name in events],
    }


def _write_run(experiment_dir, run_name: str, records: list[dict], cameras: tuple[str, ...] = ("wrist_cam",)):
    """Write one Run sub-directory holding results and a video per (episode, camera)."""
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


def test_progress_fraction_uses_recorded_overall_score():
    # overall_score is recorded already normalized to [0, 1], so it is used directly.
    episode = _episode({"progress": {"overall_score": 0.5}})

    assert episode.progress_fraction == 0.5


def test_progress_fraction_clamps_out_of_range_overall_score():
    episode = _episode({"progress": {"overall_score": 1.5}})

    assert episode.progress_fraction == 1.0


def test_progress_fraction_is_none_without_recorded_progress():
    episode = _episode({"success": True})

    assert episode.progress_fraction is None


def test_funnel_counts_criteria_instances_rather_than_events():
    episode = _episode({
        "success": False,
        "progress": _progress(
            {
                "subtask_0/pick": "object_on_destination(force_threshold=0.1)",
                "subtask_1/pick": "object_is_above_height(object_name='lemon')",
            },
            [
                ("subtask_0/pick", 0, "objects_settled"),
                ("subtask_0/pick", 0, "objects_settled"),
                ("subtask_1/pick", 0, "objects_settled"),
                ("subtask_0/pick", 1, "object_is_above_height(object_name='lemon')"),
            ],
            score=0.5,
        ),
    })
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    assert len(job.funnels) == 1
    assert job.funnels[0].num_instances == 2
    assert [(stage.index, stage.name, stage.num_reached) for stage in job.funnels[0].stages] == [
        (0, "objects_settled", 2),
        (1, "object_is_above_height", 1),
    ]


def test_criteria_list_predicates_the_episode_never_reached():
    complete = _episode({
        "success": True,
        "progress": _progress(
            {"pick_and_place": None},
            [
                ("pick_and_place", 0, "objects_settled"),
                ("pick_and_place", 1, "object_is_above_height(object_name='banana')"),
                ("pick_and_place", 2, "object_on_destination(force_threshold=0.1)"),
            ],
            score=1.0,
        ),
    })
    stalled = _episode(
        {
            "success": False,
            "progress": {
                "overall_score": 0.33,
                "criteria_by_name": {
                    "pick_and_place": {
                        "score": 0.33,
                        "is_complete": False,
                        "total_sequences": 1,
                        "active_predicates": {"default_sequence": "object_is_above_height(object_name='banana')"},
                    }
                },
                "events": [_event("pick_and_place", "default_sequence", 0, "objects_settled", step=7)],
            },
        },
        episode=1,
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[complete, stalled])

    criteria = job.criteria_for(stalled)[0]
    assert criteria.num_triggered == 1
    assert [(signal.name, signal.triggered, signal.blocked) for signal in criteria.signals] == [
        ("objects_settled", True, False),
        ("object_is_above_height", False, True),
        ("object_on_destination", False, False),
    ]
    assert criteria.signals[0].step == 7


def test_temporal_predicates_keep_distinct_report_labels_and_recorded_details():
    resting_requirement = "TrueForConsecutiveStepsCfg(objects_below_velocity_thresholds, required_steps=10)"
    placement_requirement = "TrueForConsecutiveStepsCfg(object_on_destination(force_threshold=0.1), required_steps=5)"
    completed_progress = _progress(
        {"pick_and_place": None},
        [
            ("pick_and_place", 0, resting_requirement),
            ("pick_and_place", 1, placement_requirement),
        ],
        score=1.0,
    )
    stalled_progress = _progress({"pick_and_place": resting_requirement}, [], score=0.0)
    complete = _episode({"success": True, "progress": completed_progress})
    stalled = _episode({"success": False, "progress": stalled_progress}, episode=1)
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[complete, stalled])

    assert [(stage.name, stage.num_reached) for stage in job.funnels[0].stages] == [
        ("objects_below_velocity_thresholds", 1),
        ("object_on_destination", 1),
    ]
    assert [(signal.name, signal.blocked) for signal in job.criteria_for(stalled)[0].signals] == [
        ("objects_below_velocity_thresholds", True),
        ("object_on_destination", False),
    ]
    assert [signal.detail for signal in job.criteria_for(complete)[0].signals] == [
        resting_requirement,
        placement_requirement,
    ]


def test_compatible_subtask_criteria_are_coalesced_into_one_family():
    episode = _episode({
        "progress": {
            "overall_score": 1.0,
            "criteria_by_name": {
                "subtask_0/pick_and_place": {
                    "score": 1.0,
                    "is_complete": True,
                    "total_sequences": 1,
                    "active_predicates": {"default_sequence": None},
                },
                "subtask_1/pick_and_place": {
                    "score": 0.33,
                    "is_complete": False,
                    "total_sequences": 1,
                    "active_predicates": {"default_sequence": "object_is_above_height(object_name='banana')"},
                },
            },
            "events": [
                _event("subtask_0/pick_and_place", "default_sequence", 0, "objects_settled"),
                _event("subtask_1/pick_and_place", "default_sequence", 0, "objects_settled"),
            ],
        }
    })
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    criteria_sets = job.criteria_for(episode)
    assert [criteria.family for criteria in criteria_sets] == ["pick_and_place", "pick_and_place"]
    assert [funnel.name for funnel in job.funnels] == ["pick_and_place"]


def test_conflicting_subtask_sequences_stay_split_and_report_an_issue():
    episode = _episode({
        "progress": {
            "criteria_by_name": {
                "subtask_0/pick": {
                    "score": 0.0,
                    "is_complete": False,
                    "total_sequences": 1,
                    "active_predicates": {"default_sequence": "second_predicate"},
                },
                "subtask_1/pick": {
                    "score": 0.0,
                    "is_complete": False,
                    "total_sequences": 1,
                    "active_predicates": {"default_sequence": "second_predicate"},
                },
            },
            "events": [
                _event("subtask_0/pick", "default_sequence", 0, "first_predicate"),
                _event("subtask_1/pick", "default_sequence", 0, "other_predicate"),
            ],
        }
    })
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    assert any("conflicting predicate sequences" in issue.message for issue in job.issues)
    criteria_sets = job.criteria_for(episode)
    assert [criteria.family for criteria in criteria_sets] == ["subtask_0/pick", "subtask_1/pick"]


def test_unknown_active_predicates_are_renderable_without_inventing_sequence_indices():
    episode = _episode({
        "progress": {
            "criteria_by_name": {
                "pick": {
                    "score": 0.0,
                    "is_complete": False,
                    "total_sequences": 1,
                    "active_predicates": {"default": "never_seen_predicate(arg=1)"},
                }
            },
            "events": [],
        }
    })
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    assert [(funnel.name, funnel.num_instances, funnel.stages) for funnel in job.funnels] == [("pick", 1, [])]
    assert not job.has_incomplete_sequence_data
    criteria = job.criteria_for(episode)[0]
    assert criteria.signals == []
    assert criteria.blocked_predicates == ["never_seen_predicate"]


def test_missing_sequence_names_are_flagged_without_assigning_attempts():
    failed = _episode({"progress": {"criteria_by_name": {"reach": {"total_sequences": 1}}, "events": []}})
    succeeded = _episode(
        {
            "progress": {
                "criteria_by_name": {"reach": {"total_sequences": 1}},
                "events": [_event("reach", "default_sequence", 0, "arrive")],
            }
        },
        episode=1,
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[failed, succeeded])

    assert job.has_incomplete_sequence_data
    assert job.funnels[0].num_instances == 1
    assert job.funnels[0].stages[0].num_reached == 1


def test_partial_sequence_names_are_flagged():
    episode = _episode(
        {"progress": {"criteria_by_name": {"reach": {"total_sequences": 2, "active_predicates": {"left": "arrive"}}}}}
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    assert job.has_incomplete_sequence_data


def test_sequences_keep_independent_funnels_and_event_steps():
    episode = _episode({
        "progress": {
            "criteria_by_name": {
                "reach": {
                    "score": 0.5,
                    "is_complete": False,
                    "total_sequences": 2,
                    "active_predicates": {"left": None, "right": "arrive"},
                }
            },
            "events": [
                _event("reach", "left", 0, "found", step=3),
                _event("reach", "right", 0, "found", step=7),
                _event("reach", "left", 1, "arrive", step=9),
            ],
        }
    })
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    assert [(funnel.name, funnel.num_instances) for funnel in job.funnels] == [("reach/left", 1), ("reach/right", 1)]
    assert [[stage.num_reached for stage in funnel.stages] for funnel in job.funnels] == [[1, 1], [1]]
    assert [(signal.name, signal.step) for signal in job.criteria_for(episode)[0].signals] == [
        ("left/found", 3),
        ("left/arrive", 9),
        ("right/found", 7),
    ]


def test_sequence_funnels_count_episodes_where_the_sequence_emitted_no_event():
    first = _episode({
        "progress": {
            "criteria_by_name": {"reach": {"active_predicates": {"left": None, "right": "arrive"}}},
            "events": [_event("reach", "left", 0, "arrive")],
        }
    })
    second = _episode(
        {
            "progress": {
                "criteria_by_name": {"reach": {"active_predicates": {"left": "arrive", "right": None}}},
                "events": [_event("reach", "right", 0, "arrive")],
            }
        },
        episode=1,
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[first, second])

    assert [(funnel.name, funnel.num_instances, funnel.stages[0].num_reached) for funnel in job.funnels] == [
        ("reach/left", 2, 1),
        ("reach/right", 2, 1),
    ]


def test_wait_in_an_eventless_sibling_sequence_is_listed_without_an_index():
    left_done = _episode({
        "progress": {
            "criteria_by_name": {"reach": {"active_predicates": {"left": None, "right": "arrive"}}},
            "events": [_event("reach", "left", 0, "arrive")],
        }
    })
    stalled = _episode(
        {"progress": {"criteria_by_name": {"reach": {"active_predicates": {"left": "arrive", "right": "arrive"}}}}},
        episode=1,
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[left_done, stalled])

    # The right sequence never emitted an event, so its funnel has no stages and its wait has no index.
    assert [(funnel.name, len(funnel.stages)) for funnel in job.funnels] == [("reach/left", 1), ("reach/right", 0)]
    assert job.criteria_for(left_done)[0].blocked_predicates == ["right/arrive"]
    stalled_criteria = job.criteria_for(stalled)[0]
    assert [(signal.name, signal.blocked) for signal in stalled_criteria.signals] == [("left/arrive", True)]
    assert stalled_criteria.blocked_predicates == ["right/arrive"]


def test_waiting_predicates_are_matched_within_their_sequence():
    # The two sequences reach the same predicates in opposite order, so a wait matched against
    # another sequence's predicates would block every signal.
    complete = _episode({
        "progress": {
            "criteria_by_name": {"reach": {"active_predicates": {"left": None, "right": None}}},
            "events": [
                _event("reach", "left", 0, "found"),
                _event("reach", "left", 1, "arrive"),
                _event("reach", "right", 0, "arrive"),
                _event("reach", "right", 1, "found"),
            ],
        }
    })
    stalled = _episode(
        {"progress": {"criteria_by_name": {"reach": {"active_predicates": {"left": "found", "right": "arrive"}}}}},
        episode=1,
    )
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[complete, stalled])

    stalled_criteria = job.criteria_for(stalled)[0]
    assert [(signal.name, signal.blocked) for signal in stalled_criteria.signals] == [
        ("left/found", True),
        ("left/arrive", False),
        ("right/arrive", True),
        ("right/found", False),
    ]
    assert stalled_criteria.blocked_predicates == []


def test_subtask_criteria_coalesce_when_each_sequence_agrees():
    episode = _episode({
        "progress": {
            "criteria_by_name": {
                "subtask_0/reach": {"active_predicates": {"left": None, "right": None}},
                "subtask_1/reach": {"active_predicates": {"left": None}},
            },
            "events": [
                _event("subtask_0/reach", "left", 0, "found"),
                _event("subtask_0/reach", "right", 0, "arrive"),
                _event("subtask_1/reach", "left", 0, "found"),
            ],
        }
    })
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=[episode])

    assert not job.issues
    assert [(funnel.name, funnel.num_instances) for funnel in job.funnels] == [("reach/left", 2), ("reach/right", 1)]
    assert [signal.name for signal in job.criteria_for(episode)[1].signals] == ["left/found"]


def test_conflicting_predicate_names_identify_the_sequence():
    episodes = [
        _episode(
            {
                "progress": {
                    "criteria_by_name": {"reach": {"active_predicates": {"left": None}}},
                    "events": [_event("reach", "left", 0, predicate_name)],
                }
            },
            episode=index,
        )
        for index, predicate_name in enumerate(("arrive", "settle"))
    ]
    job = JobSummary(name="run", task="t", policy="p", cameras=[], episodes=episodes)

    assert [issue.message for issue in job.issues] == [
        "completion criteria family 'reach' sequence 'left' has multiple predicate names at index 0:"
        " ['arrive', 'settle']"
    ]


def test_outcome_disagreeing_with_progress_is_detected():
    complete_but_failed = _episode({"success": False, "progress": {"all_complete": True, "overall_score": 2.0}})
    incomplete_but_passed = _episode({"success": True, "progress": {"all_complete": False, "overall_score": 0.5}})
    agreeing = _episode({"success": True, "progress": {"all_complete": True}})
    no_progress_block = _episode({"success": True})

    assert complete_but_failed.outcome_disagrees_with_progress
    assert incomplete_but_passed.outcome_disagrees_with_progress
    assert not agreeing.outcome_disagrees_with_progress
    assert no_progress_block.all_criteria_complete is None
    assert not no_progress_block.outcome_disagrees_with_progress


def test_summary_mean_progress_averages_scored_episodes_across_runs(tmp_path):
    def record(episode: int, score: float | None) -> dict:
        entry = {"env_id": 0, "episode_in_env": episode, "success": False}
        return entry if score is None else {**entry, "progress": {"overall_score": score}}

    _write_run(tmp_path, "banana_pi0", [record(0, 0.25), record(1, 0.75)])
    _write_run(tmp_path, "bowl_pi0", [record(0, 0.5), record(1, None)])
    _write_run(tmp_path, "banana_cosmos", [record(0, 1.0)])

    summary = build_experiment_summary(tmp_path, "Report")

    # The unscored episode is left out of both the run mean and the aggregates above it.
    assert summary.tasks[1].job_for_policy("pi0").mean_progress == 0.5
    assert summary.mean_progress_for_policy("pi0") == 0.5
    assert summary.mean_progress_for_policy("cosmos") == 1.0
    assert summary.overall_mean_progress == 0.625


def test_summary_mean_progress_is_none_without_recorded_progress(tmp_path):
    _write_run(tmp_path, "banana_pi0", [{"env_id": 0, "episode_in_env": 0, "success": True}])

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.overall_mean_progress is None
    assert summary.mean_progress_for_policy("pi0") is None


def test_summary_groups_sparse_runs_by_repeated_policy_tokens(tmp_path):
    _write_run(tmp_path, "banana_pi0", [{"env_id": 0, "episode_in_env": 0, "success": True}])
    _write_run(tmp_path, "banana_cosmos", [{"env_id": 0, "episode_in_env": 0, "success": False}])
    _write_run(tmp_path, "bowl_cosmos", [{"env_id": 0, "episode_in_env": 0, "success": True}])

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.grouping_source == "run_names"
    assert [task.name for task in summary.tasks] == ["banana", "bowl"]
    assert summary.policies == ["cosmos", "pi0"]
    assert summary.tasks[1].job_for_policy("pi0") is None


def test_infer_labels_recovers_explicit_multi_token_policy_suffixes():
    labels, source = _infer_task_and_policy_labels_with_source(
        ["banana_pi0_remote", "banana_cosmos_remote", "bowl_cosmos_remote"],
        policy_suffixes=("pi0_remote", "cosmos_remote"),
    )

    assert source == "policy_suffixes"
    assert labels["banana_pi0_remote"] == ("banana", "pi0_remote")
    assert labels["bowl_cosmos_remote"] == ("bowl", "cosmos_remote")


def test_explicit_policy_suffixes_do_not_partially_group_runs():
    assert _infer_labels_from_explicit_suffixes(["task_pi0", "task_openvla", "other_openvla"], ("pi0",)) is None

    labels, source = _infer_task_and_policy_labels_with_source(
        ["task_pi0", "task_openvla", "other_openvla"],
        policy_suffixes=("pi0",),
    )

    assert source == "run_names"
    assert labels == {
        "task_pi0": ("task", "pi0"),
        "task_openvla": ("task", "openvla"),
        "other_openvla": ("other", "openvla"),
    }


def test_partial_explicit_policy_suffixes_fall_back_to_run_name_grouping(tmp_path):
    _write_run(tmp_path, "task_pi0", [{"env_id": 0, "episode_in_env": 0, "success": True}])
    _write_run(tmp_path, "task_openvla", [{"env_id": 0, "episode_in_env": 0, "success": False}])
    _write_run(tmp_path, "other_openvla", [{"env_id": 0, "episode_in_env": 0, "success": True}])

    summary = build_experiment_summary(tmp_path, "Report", policy_suffixes=("pi0",))

    assert summary.grouping_source == "run_names"
    assert summary.policies == ["openvla", "pi0"]


def test_default_grouping_rejects_repeated_task_words_as_policy_names():
    labels, source = _infer_task_and_policy_labels_with_source(["small_cube_pick", "large_cube_pick", "banana_in_bowl"])

    assert labels is None
    assert source == "none"


def test_explicit_policy_suffix_can_group_a_single_run(tmp_path):
    _write_run(tmp_path, "banana_pi0", [{"env_id": 0, "episode_in_env": 0, "success": True}])

    summary = build_experiment_summary(tmp_path, "Report", policy_suffixes=("pi0",))

    assert summary.grouping_source == "policy_suffixes"
    assert summary.tasks[0].name == "banana"
    assert summary.tasks[0].job_for_policy("pi0") is not None


def test_summary_leaves_runs_ungrouped_when_no_labels_can_be_established(tmp_path):
    _write_run(tmp_path, "solo_run", [{"env_id": 0, "episode_in_env": 0, "success": True}])

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.grouping_source == "none"
    assert summary.is_grouped is False
    assert [task.name for task in summary.tasks] == ["solo_run"]


def test_summary_excludes_runs_whose_process_failed_with_string_status(tmp_path):
    _write_run(tmp_path, "good_pi0", [{"env_id": 0, "episode_in_env": 0, "success": True}])
    _write_run(tmp_path, "broken_pi0", [{"env_id": 0, "episode_in_env": 0, "success": False}])

    summary = build_experiment_summary(
        tmp_path,
        "Report",
        [RunExecutionReport(run_name="broken_pi0", status="failed", process_exit_code=17)],
    )

    assert [job.name for job in summary.jobs] == ["good_pi0"]
    assert summary.num_episodes == 1


def test_summary_pairs_videos_with_records_and_preserves_missing_media(tmp_path):
    run_dir = _write_run(
        tmp_path,
        "banana_pi0",
        [{"env_id": 0, "episode_in_env": 0, "success": True}],
        cameras=("wrist_cam", "front_cam"),
    )
    (run_dir / format_episode_video_filename("robot-cam-rebuild0", 0, "front_cam", 0)).unlink()

    summary = build_experiment_summary(tmp_path, "Report")

    job = summary.jobs[0]
    assert job.cameras == ["wrist_cam"]
    assert job.num_videos == 1
    assert job.episodes[0].success is True
    assert "wrist_cam" in job.episodes[0].video_by_camera


def test_summary_pairs_nonzero_rebuild_results_with_matching_videos(tmp_path):
    run_dir = tmp_path / "banana_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild2.jsonl").write_text(
        json.dumps({"env_id": 0, "episode_in_env": 4, "success": True}) + "\n", encoding="utf-8"
    )
    video_name = format_episode_video_filename("robot-cam-rebuild2", 0, "wrist_cam", 4)
    (run_dir / video_name).write_bytes(b"")

    summary = build_experiment_summary(tmp_path, "Report")

    episode = summary.jobs[0].episodes[0]
    assert episode.rebuild_index == 2
    assert episode.video_by_camera == {"wrist_cam": f"banana_pi0/{video_name}"}


def test_video_only_runs_remain_visible(tmp_path):
    run_dir = tmp_path / "video_only_pi0"
    run_dir.mkdir()
    (run_dir / format_episode_video_filename("robot-cam-rebuild0", 2, "wrist_cam", 5)).write_bytes(b"")

    summary = build_experiment_summary(tmp_path, "Report")

    job = summary.jobs[0]
    assert job.num_episodes == 1
    assert job.episodes[0].success is None
    assert job.num_videos == 1


def test_jsonl_only_runs_remain_visible(tmp_path):
    run_dir = tmp_path / "json_only_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        json.dumps({"env_id": 0, "episode_in_env": 0, "success": True}) + "\n", encoding="utf-8"
    )

    summary = build_experiment_summary(tmp_path, "Report")

    job = summary.jobs[0]
    assert job.num_episodes == 1
    assert job.num_videos == 0


def test_nested_directory_named_report_is_not_skipped_as_generated_output(tmp_path):
    run_dir = tmp_path / "suite" / "report"
    run_dir.mkdir(parents=True)
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        json.dumps({"env_id": 0, "episode_in_env": 0, "success": True}) + "\n", encoding="utf-8"
    )

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.num_episodes == 1
    assert summary.jobs[0].name == "suite/report"


def test_malformed_and_incomplete_records_are_reported_and_skipped(tmp_path):
    run_dir = tmp_path / "bad_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        "\n".join([
            "{bad json",
            json.dumps({"env_id": 0, "success": True}),
            json.dumps({"env_id": 1, "episode_in_env": 0, "success": False}),
        ]),
        encoding="utf-8",
    )

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.num_episodes == 1
    assert any("invalid JSON" in issue.message for issue in summary.issues)
    assert any("episode_in_env" in issue.message for issue in summary.issues)


def test_rank_result_files_do_not_overwrite_each_other(tmp_path):
    for rank, success in ((0, True), (1, False)):
        (tmp_path / f"episode_results_rank{rank}.jsonl").write_text(
            json.dumps({"env_id": 0, "episode_in_env": 0, "success": success}) + "\n", encoding="utf-8"
        )

    summary = build_experiment_summary(tmp_path, "Report")

    assert summary.num_episodes == 2
    assert sorted(episode.success for episode in summary.jobs[0].episodes) == [False, True]


def test_run_status_normalizes_enum_like_values():
    class Status:
        value = "FAILED"

    assert normalize_run_status(Status()) == "failed"


def test_temporal_episode_summary_handles_missing_and_malformed_records():
    assert _episode().consecutive_step_summaries == []
    assert _episode({"progress": {"criteria_by_name": {"task": {"score": 0.5}}}}).consecutive_step_summaries == []
    valid = {
        "predicate_index": 0,
        "predicate_name": "TrueForConsecutiveStepsCfg(settled, required_steps=10)",
        "consecutive_steps": 6,
        "required_steps": 10,
        "status": "active",
    }
    assert _episode({"progress": {"consecutive_step_summary": None}}).consecutive_step_summaries == []
    valid.update(criteria_name="task", sequence_name="left")
    episode = _episode({
        "progress": {
            "consecutive_step_summary": [
                None,
                {},
                {**valid, "required_steps": 0},
                {**valid, "consecutive_steps": 11},
                valid,
                {
                    **valid,
                    "sequence_name": "right",
                    "consecutive_steps": 10,
                    "status": "completed",
                    "first_satisfied_step": 12,
                },
            ]
        }
    })
    summaries = episode.consecutive_step_summaries
    assert [(item.criteria_name, item.sequence_name, item.predicate_index) for item in summaries] == [
        ("task", "left", 0),
        ("task", "right", 0),
    ]
    assert summaries[0].predicate_name == "settled"
    assert summaries[0].consecutive_steps == 6
    assert summaries[0].first_satisfied_step is None
    assert summaries[1].first_satisfied_step == 12


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("field", ["consecutive_steps", "required_steps", "predicate_index", "first_satisfied_step"])
def test_consecutive_step_summary_handles_nonfinite_numbers(field, value):
    valid = {
        "predicate_index": 0,
        "predicate_name": "settled",
        "consecutive_steps": 6,
        "required_steps": 10,
        "status": "active",
        "first_satisfied_step": None,
    }
    episode = _episode({"progress": {"consecutive_step_summary": [{**valid, field: value}, valid]}})
    summaries = episode.consecutive_step_summaries
    # An invalid optional completion step is unknown; invalid required fields discard the entry.
    assert len(summaries) == (2 if field == "first_satisfied_step" else 1)
    assert all(summary.first_satisfied_step is None for summary in summaries)
    assert all(summary.consecutive_steps == 6 for summary in summaries)
