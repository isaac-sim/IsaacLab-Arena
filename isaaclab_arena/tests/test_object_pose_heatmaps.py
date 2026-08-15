# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np

from isaaclab_arena_examples.sensitivity_analysis.plot_object_pose_heatmaps import (
    align_env_positions,
    bin_xy,
    generate_heatmaps,
    grid_edges,
)


def test_bin_xy_counts_and_normalized_success_rates():
    xy = np.asarray([[0.01, 0.01], [0.02, 0.01], [0.07, 0.07]])
    successes = np.asarray([True, False, True])
    edges = np.asarray([0.0, 0.05, 0.10])

    counts, rates = bin_xy(xy, edges, edges, successes)

    assert rates is not None
    assert counts.tolist() == [[2.0, 0.0], [0.0, 1.0]]
    assert rates[0, 0] == 0.5
    assert rates[1, 1] == 1.0
    assert np.isnan(rates[0, 1])
    assert np.isnan(rates[1, 0])


def test_generate_heatmaps_from_episode_results(tmp_path):
    task_dir = tmp_path / "task_a"
    task_dir.mkdir()
    records = [
        {
            "job_name": "task_a",
            "success": True,
            "initial_reset_positions": {"object": [0.01, 0.01, 0.1]},
            "initial_rest_positions": {"object": [0.02, 0.01, 0.05]},
        },
        {
            "job_name": "task_a",
            "success": False,
            "initial_reset_positions": {"object": [0.08, 0.07, 0.1]},
            "initial_rest_positions": {"object": [0.07, 0.07, 0.05]},
        },
    ]
    results_path = task_dir / "episode_results_rebuild0.jsonl"
    results_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    written = generate_heatmaps(tmp_path, tmp_path / "plots", grid_size_m=0.05)

    assert len(written) == 1
    assert written[0].name == "task_a__object.png"
    assert written[0].is_file()
    assert written[0].stat().st_size > 0


def test_grid_edges_are_aligned_to_requested_cell_size():
    edges = grid_edges(np.asarray([-0.021, 0.081]), grid_size_m=0.05)

    assert np.allclose(edges, [-0.05, 0.0, 0.05, 0.10])


def test_align_env_positions_removes_replication_offsets():
    records = [
        {
            "env_id": 0,
            "initial_reset_positions": {"object": [30.4, -14.8, 0.1]},
            "initial_rest_positions": {"object": [30.5, -14.7, 0.1]},
        },
        {
            "env_id": 1,
            "initial_reset_positions": {"object": [30.4, 15.2, 0.1]},
            "initial_rest_positions": {"object": [30.5, 15.3, 0.1]},
        },
    ]

    aligned = align_env_positions(records, env_spacing_m=30.0)
    env0 = aligned[0]["initial_reset_positions"]["object"]
    env1 = aligned[1]["initial_reset_positions"]["object"]

    assert np.allclose(env0[:2], env1[:2])
    assert np.allclose(env0[:2], [-0.05, -0.05])
    assert env0[2] == 0.1
