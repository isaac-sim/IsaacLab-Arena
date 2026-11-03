# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report completed event measurements without changing legacy occurrence displays."""

import copy
import json

from isaaclab_arena.visualization.report import build_report
from isaaclab_arena.visualization.report_data import EpisodeIdentity, EpisodeSummary
from isaaclab_arena.visualization.report_render import _render_tracked_event_occurrences


def test_event_measurements_render_inside_occurrences_and_escape_values(tmp_path):
    run = tmp_path / "observations"
    run.mkdir()
    record = {
        "env_id": 0,
        "episode_in_env": 0,
        "success": False,
        "progress": {
            "tracked_event_names": ["arrival<script>", "never_seen"],
            "tracked_events": [
                {
                    "name": "arrival<script>",
                    "step": 4,
                    "details": {"height_m": 0.3, "note<img>": "<script>alert(1)</script>"},
                },
                {"name": "arrival<script>", "step": 9, "details": {"height_m": 0.7}},
            ],
        },
    }
    (run / "episode_results_rebuild0.jsonl").write_text(json.dumps(record) + "\n")
    build_report(tmp_path)
    pages = list((tmp_path / "report").glob("job_*.html"))
    assert len(pages) == 1
    text = pages[0].read_text()
    section = text.split("<summary>Tracked occurrences</summary>", 1)[1].split("</details>", 1)[0]
    assert "Event measurements" in section
    assert "arrival&lt;script&gt; (2)" in section
    assert "never_seen (0)" in section
    assert "<td>4</td><td>height_m</td><td>0.3</td>" in section
    assert "<td>9</td><td>height_m</td><td>0.7</td>" in section
    assert "note&lt;img&gt;" in section
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in section
    assert "arrival<script>" not in text
    assert "<script>alert(1)</script>" not in text


def test_empty_or_missing_details_preserve_legacy_occurrence_html():
    episode = EpisodeSummary(
        identity=EpisodeIdentity("episode_results_rebuild0.jsonl", 0, 0, 0),
        episode_index=0,
        video_by_camera={"wrist_cam": "episode.mp4"},
        record={
            "episode_length": 20,
            "progress": {
                "tracked_event_names": ["arrival", "never_seen"],
                "tracked_events": [{"name": "arrival", "step": 4}],
                "control_step_dt": 0.02,
            },
        },
    )
    baseline = _render_tracked_event_occurrences(episode)
    assert "Event measurements" not in baseline
    assert 'data-seek-time="0.06"' in baseline
    assert "step 4 (0.080 s)" in baseline
    assert "never_seen (0)" in baseline
    for details in ({}, None, [], "invalid"):
        changed = copy.deepcopy(episode)
        changed.record["progress"]["tracked_events"][0]["details"] = details
        assert _render_tracked_event_occurrences(changed) == baseline


def test_measurements_use_stable_json_and_the_timeline_step_parser():
    episode = EpisodeSummary(
        identity=EpisodeIdentity("episode_results_rebuild0.jsonl", 0, 0, 0),
        episode_index=0,
        video_by_camera={},
        record={
            "progress": {
                "tracked_events": [
                    {"name": "arrival", "step": "4", "details": {"z": {"b": 2, "a": 1}, "a": [True, None]}}
                ]
            }
        },
    )
    baseline = _render_tracked_event_occurrences(episode)
    assert "<td>4</td><td>a</td><td>[true, null]</td>" in baseline
    assert "{&quot;a&quot;: 1, &quot;b&quot;: 2}" in baseline
    changed = copy.deepcopy(episode)
    changed.record["progress"]["tracked_events"][0]["details"] = {"a": [True, None], "z": {"a": 1, "b": 2}}
    assert _render_tracked_event_occurrences(changed) == baseline
