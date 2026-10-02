# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check cache diagnostics without Docker or simulation dependencies."""

import io
import json
import unittest

from report_build_cache import summarize


def build_events(*vertices):
    """Encode BuildKit progress events as JSON lines."""
    return "".join(json.dumps({"vertexes": [vertex]}) + "\n" for vertex in vertices)  # codespell:ignore vertexes


class TestBuildCacheReport(unittest.TestCase):
    def test_cached_step_and_normal_final_newline(self):
        stream = build_events(
            {
                "digest": "cache-import",
                "name": "importing cache manifest from example:cache",
                "completed": "2026-09-18T10:00:00Z",
            },
            {
                "digest": "install",
                "name": "[deps 1/1] RUN install",
                "cached": True,
                "started": "2026-09-18T10:00:00Z",
                "completed": "2026-09-18T10:00:02Z",
            },
        )
        report = summarize(io.StringIO(stream))
        assert "example:cache`: imported" in report
        assert "1/1 Dockerfile steps CACHED" in report
        assert "| CACHED | 2.0 |" in report
        assert "Warning:" not in report

    def test_repeated_step_preserves_cache_hit_and_download_time(self):
        stream = build_events(
            {
                "digest": "copy",
                "name": "[dev 1/1] COPY source /app",
                "cached": True,
                "started": "2026-09-18T10:00:00Z",
                "completed": "2026-09-18T10:00:01Z",
            },
            {
                "digest": "copy",
                "name": "[dev 1/1] COPY source /app",
                "started": "2026-09-18T10:00:10Z",
                "completed": "2026-09-18T10:00:20Z",
            },
        )
        report = summarize(io.StringIO(stream))
        assert "1/1 Dockerfile steps CACHED" in report
        assert "| CACHED | 20.0 |" in report

    def test_mixed_output_preserves_steps_and_warns(self):
        step = build_events({"digest": "install", "name": "[deps 1/1] RUN install", "cached": True})
        stream = "\nerror: no build record\n" + step + "{broken\nnull\n\n"
        report = summarize(io.StringIO(stream))
        assert "1/1 Dockerfile steps CACHED" in report
        assert "skipped 3 non-JSON or non-object records" in report
        assert "report may be incomplete" in report
        assert "build-cache.jsonl" in report

    def test_error_only_output_explains_missing_diagnostics(self):
        report = summarize(io.StringIO("error: no build record\n"))
        assert "Warning:" in report
        assert "No Dockerfile steps found" in report
        assert "No registry cache import was recorded" in report

    def test_failed_step_is_not_reported_as_cached(self):
        stream = build_events({
            "digest": "install",
            "name": "[deps 1/1] RUN install",
            "cached": True,
            "error": "failed to download cached layer",
        })
        report = summarize(io.StringIO(stream))
        assert "0/1 Dockerfile steps CACHED" in report
        assert "| FAILED |" in report


if __name__ == "__main__":
    unittest.main()
