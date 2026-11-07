# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SimulationApp-free scene inspector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py"


def _run_scene_inspector(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_prim_tree_command_writes_text_file(tmp_path: Path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """\
#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
)

def Xform "Asset"
{
    def Cube "Counter" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
    }
}
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "prim_tree.txt"

    result = _run_scene_inspector(
        "prim-tree",
        "--usd",
        str(usd_path),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    json.loads(result.stdout)
    output = output_path.read_text(encoding="utf-8")
    assert output.startswith("USD PRIM TREE:\nCounter ")


def test_prim_tree_command_supports_prim_scope_and_json(tmp_path: Path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """\
#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset"
{
    def Xform "Selected"
    {
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
    }
    def Cube "Excluded" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "prim_tree.json"
    result = _run_scene_inspector(
        "prim-tree",
        "--usd",
        str(usd_path),
        "--prim",
        "Selected",
        "--format",
        "json",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    json.loads(result.stdout)
    records = json.loads(output_path.read_text())
    assert [record["relative_path"] for record in records] == ["Collider"]
