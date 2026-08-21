# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np

from isaaclab_arena_examples.sensitivity_analysis.plot_object_pose_heatmaps import (
    bin_xy,
    generate_heatmaps,
    grid_edges,
    to_env_local_positions,
)


def test_bin_xy_counts_support_optional_weights():
    xy = np.asarray([[0.01, 0.01], [0.02, 0.01], [0.07, 0.07]])
    successes = np.asarray([True, False, True])
    edges = np.asarray([0.0, 0.05, 0.10])

    counts = bin_xy(xy, edges, edges)
    success_counts = bin_xy(xy, edges, edges, successes)

    assert counts.tolist() == [[2.0, 0.0], [0.0, 1.0]]
    assert success_counts.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_generate_heatmaps_from_episode_results(tmp_path):
    task_dir = tmp_path / "task_a"
    task_dir.mkdir()
    records = [
        {
            "job_name": "task_a",
            "env_origin": [0.0, 0.0, 0.0],
            "success": True,
            "reset_positions": {"object": [0.01, 0.01, 0.1]},
            "settled_positions": {"object": [0.02, 0.01, 0.05]},
        },
        {
            "job_name": "task_a",
            "env_origin": [0.0, 0.0, 0.0],
            "success": False,
            "reset_positions": {"object": [0.08, 0.07, 0.1]},
            "settled_positions": {"object": [0.07, 0.07, 0.05]},
        },
    ]
    results_path = task_dir / "episode_results_rebuild0.jsonl"
    results_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    written = generate_heatmaps(tmp_path, tmp_path / "plots", grid_size_m=0.05)

    assert len(written) == 1
    assert written[0].name == "task_a__object.png"
    assert written[0].is_file()
    assert written[0].stat().st_size > 0


def test_generate_heatmaps_from_arena_experiment_result_by_policy(tmp_path):
    episode = {
        "env_id": 0,
        "env_origin": [0.0, 0.0, 0.0],
        "success": True,
        "reset_positions": {"object": [0.01, 0.01, 0.1]},
        "settled_positions": {"object": [0.02, 0.01, 0.05]},
    }
    result_path = tmp_path / "arena_experiment_result.json"
    result_path.write_text(
        json.dumps({
            "runs": {
                "task_a_pi0": {
                    "status": "completed",
                    "rebuilds": [{"index": 0, "episodes": [{**episode, "job_name": "task_a_pi0"}]}],
                },
                "task_a_cosmos": {
                    "status": "completed",
                    "rebuilds": [{"index": 0, "episodes": [{**episode, "job_name": "task_a_cosmos"}]}],
                },
            }
        }),
        encoding="utf-8",
    )

    written = generate_heatmaps(
        result_path,
        tmp_path / "plots",
        grid_size_m=0.05,
        policies=["pi0", "cosmos"],
    )

    assert {path.relative_to(tmp_path / "plots").as_posix() for path in written} == {
        "pi0/task_a__object.png",
        "cosmos/task_a__object.png",
    }
    assert all(path.stat().st_size > 0 for path in written)


def test_grid_edges_are_aligned_to_requested_cell_size():
    edges = grid_edges(np.asarray([-0.021, 0.081]), grid_size_m=0.05)

    assert np.allclose(edges, [-0.05, 0.0, 0.05, 0.10])


def test_to_env_local_positions_subtracts_recorded_origins_only():
    records = [
        {
            "env_id": 0,
            "env_origin": [30.0, -15.0, 0.0],
            "reset_positions": {"object": [30.4, -14.8, 0.1]},
            "settled_positions": {"object": [30.5, -14.7, 0.1]},
        },
        {
            "env_id": 1,
            "env_origin": [30.0, 15.0, 0.0],
            "reset_positions": {"object": [30.4, 15.2, 0.1]},
            "settled_positions": {"object": [30.5, 15.3, 0.1]},
        },
    ]

    local = to_env_local_positions(records)
    reset_env0 = local[0]["reset_positions"]["object"]
    reset_env1 = local[1]["reset_positions"]["object"]
    settled_env0 = local[0]["settled_positions"]["object"]
    settled_env1 = local[1]["settled_positions"]["object"]

    assert np.allclose(reset_env0[:2], reset_env1[:2])
    assert np.allclose(reset_env0[:2], [0.4, 0.2])
    assert reset_env0[2] == 0.1
    assert np.allclose(settled_env0[:2], settled_env1[:2])
    assert np.allclose(settled_env0[:2], [0.5, 0.3])
    assert settled_env0[2] == 0.1
