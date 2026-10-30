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

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py"


def test_prim_tree_command_writes_text_file(tmp_path: Path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """\
#usda 1.0
(
    defaultPrim = "Asset"
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

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "prim-tree",
            "--usd",
            str(usd_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["status"] == "completed"
    assert response["record_count"] == 1
    assert response["output"] == str(output_path)
    output = output_path.read_text(encoding="utf-8")
    assert output.startswith("USD PRIM TREE:\n")
    assert "Counter (base " in output
    assert "bbox_center_m=[0,0,0]" in output
    assert "bbox_size_m=[2,2,2]" in output
    assert "collider(type=Cube,enabled=true,rigid_body_owner=world,approximation=None)" in output
