# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record prerequisite readiness independently from scored task progress."""

import json
from functools import partial

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_recorded_readiness_preserves_unscored_preparation(simulation_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.recording.progress_terms import record_progress_results
    from isaaclab_arena.tests.test_task_success_from_progress import (
        _controlled_predicate,
        _make_environment_and_manager,
    )

    criteria = CompletionCriteria(
        name="pick_and_place",
        prerequisites=[partial(_controlled_predicate, predicate_name="ready")],
        predicate_sequence=[
            partial(_controlled_predicate, predicate_name="lifted"),
            partial(_controlled_predicate, predicate_name="placed"),
        ],
    )
    env, manager, recorder = _make_environment_and_manager(["ready", "lifted", "placed"], success_criteria=[criteria])
    env.predicate_results["ready"][0] = False
    env.predicate_results["lifted"][:] = False
    env.episode_length_buf += 1
    manager.compute()
    recorder.record_post_step()

    for env_id, expected_readiness in enumerate((False, True)):
        recorded = json.loads(json.dumps(record_progress_results(env, env_id)))["progress"]
        recorded_criteria = recorded["criteria_by_name"]["pick_and_place"]
        assert recorded_criteria["prerequisites_met"] is expected_readiness
        assert recorded_criteria["score"] == 0
        assert not recorded_criteria["is_complete"]
        assert recorded["overall_score"] == 0
        assert recorded["events"] == []

    env.predicate_results["ready"][:] = True
    env.predicate_results["lifted"][:] = True
    env.episode_length_buf += 1
    manager.compute()
    recorder.record_post_step()
    for env_id in range(env.num_envs):
        recorded = record_progress_results(env, env_id)["progress"]
        recorded_criteria = recorded["criteria_by_name"]["pick_and_place"]
        assert recorded_criteria["prerequisites_met"]
        assert recorded_criteria["score"] == 0.5
        assert recorded["overall_score"] == 0.5
        assert [event["predicate_index"] for event in recorded["events"]] == [0]
    return True


def test_recorded_readiness_preserves_unscored_preparation():
    assert run_function_with_persistent_simulation_app(_test_recorded_readiness_preserves_unscored_preparation)
