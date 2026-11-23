# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check cache content reporting without Docker or simulation dependencies."""

import os
import tempfile
import unittest
from pathlib import Path

from report_cache_contents import describe_restore, render_report, snapshot_paths


def write(root, relative, content):
    """Write `content` to `root/relative`, creating parent directories."""
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestCacheContentsReport(unittest.TestCase):
    def test_snapshot_counts_files_and_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, "object/a.usd", "abc")
            write(root, "object/sub/b.usd", "12345")
            snapshot = snapshot_paths([root])
        self.assertEqual(snapshot["files"], 2)
        self.assertEqual(snapshot["bytes"], 8)

    def test_snapshot_of_missing_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = snapshot_paths([os.path.join(root, "missing")])
        self.assertEqual(snapshot["files"], 0)
        self.assertEqual(snapshot["bytes"], 0)

    def test_digest_changes_when_a_file_is_added(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, "a.usd", "abc")
            before = snapshot_paths([root])
            write(root, "b.usd", "abc")
            after = snapshot_paths([root])
        self.assertNotEqual(before["digest"], after["digest"])

    def test_digest_ignores_modification_time(self):
        with tempfile.TemporaryDirectory() as root:
            path = write(root, "a.usd", "abc")
            before = snapshot_paths([root])
            os.utime(path, (1_000_000, 1_000_000))
            after = snapshot_paths([root])
        self.assertEqual(before["digest"], after["digest"])

    def test_digest_changes_when_usd_version_json_content_changes_with_same_size(self):
        with tempfile.TemporaryDirectory() as root:
            path = write(root, "object/x/usd_version.json", '{"a": "1"}')
            before = snapshot_paths([root])
            path.write_text('{"a": "2"}')
            after = snapshot_paths([root])
        self.assertEqual(before["bytes"], after["bytes"])
        self.assertNotEqual(before["digest"], after["digest"])

    def test_login_and_lock_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            write(root, "a.usd", "abc")
            before = snapshot_paths([root])
            write(root, "login", "secret")
            write(root, "object/x.lock", "")
            after = snapshot_paths([root])
        self.assertEqual(before, after)

    def test_describe_restore_statuses(self):
        self.assertEqual(describe_restore("", "key"), "miss")
        self.assertEqual(describe_restore("key", "key"), "exact hit")
        self.assertEqual(describe_restore("older", "key"), "restore-key hit")

    def test_render_report_contains_name_status_size_and_changed_flag(self):
        before = {"files": 2, "bytes": 2 * 1024 * 1024, "digest": "a"}
        after = {"files": 5, "bytes": 3 * 1024 * 1024, "digest": "b"}
        report = render_report("lightwheel", "exact hit", before, after)
        self.assertTrue("lightwheel" in report)
        self.assertTrue("exact hit" in report)
        self.assertTrue("2.0 MB" in report)
        self.assertTrue("changed: yes" in report.lower())
        self.assertTrue("Files added: 3" in report)
        unchanged = render_report("lightwheel", "miss", before, before)
        self.assertTrue("changed: no" in unchanged.lower())


if __name__ == "__main__":
    unittest.main()
