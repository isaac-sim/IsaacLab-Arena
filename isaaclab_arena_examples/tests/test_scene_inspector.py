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

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "isaaclab_arena_examples/agentic_environment_generation/scene_inspector.py"


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
    assert "collider(type=Cube,enabled=true,rigid_body_owner=world,approximation=None," in output


@pytest.mark.parametrize(
    ("prim_path", "expected_paths"),
    [("Selected", ["Collider"]), ("/Asset/Selected", ["Collider"]), ("Selected/Collider", ["."])],
)
def test_prim_tree_command_supports_prim_scope_and_json(tmp_path: Path, prim_path: str, expected_paths: list[str]):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """\
#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
{
    def Xform "Selected"
    {
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
        {
            float physxCollision:contactOffset = 1.5
            float physxCollision:restOffset = 0.5
        }
    }
    def Cube "Excluded" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
}
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "prim_tree.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "prim-tree",
            "--usd",
            str(usd_path),
            "--prim",
            prim_path,
            "--format",
            "json",
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
    assert response["inputs"]["prim"] == prim_path
    assert response["format"] == "json"
    assert response["record_count"] == len(expected_paths)
    records = json.loads(output_path.read_text())
    assert [record["relative_path"] for record in records] == expected_paths
    assert records[0]["object_type"] == "base"
    assert records[0]["collider"]["type_name"] == "Cube"
    assert records[0]["collider"]["rigid_body_owner_path"] == "/Asset"
    assert records[0]["collider"]["contact_offset_m"] == pytest.approx(0.015)
    assert records[0]["collider"]["rest_offset_m"] == pytest.approx(0.005)
    assert records[0]["bounding_box_root"]["size_m"] == pytest.approx((0.02, 0.02, 0.02))
