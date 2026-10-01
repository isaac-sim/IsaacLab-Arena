# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter recording through the unified placement CLI."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

SCRIPT = Path(TestConstants.scripts_dir) / "record_placement_layouts.py"


@pytest.mark.with_subprocess
@pytest.mark.parametrize("preset", ["physx", "newton"])
def test_cli_generates_scene_cache(tmp_path, preset):
    import yaml

    from isaaclab_arena.tests.test_settled_placement import _write_scene

    output = tmp_path / "episodes.jsonl"
    source = tmp_path / "scene.yaml"
    _write_scene(source)
    # The support instance name differs from its registry name. The reference must
    # resolve beneath that instance, including for double-digit environment IDs.
    (tmp_path / "table.usda").write_text("""#usda 1.0
(
    defaultPrim = "Body"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    bool physics:kinematicEnabled = true
    float physics:mass = 1
    def Xform "surface" {
        double3 xformOp:translate = (0, 0, 0)
        quatf xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        def Cube "geometry" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 1
            double3 xformOp:scale = (0.8, 0.8, 0.04)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }
    }
}
""")
    data = yaml.safe_load(source.read_text())
    data["background"]["params"]["prim_path"] = "{ENV_REGEX_NS}/fixtures/custom_table"
    data["object_references"] = [{"id": "surface", "parent_id": "table", "prim_path": "surface", "object_type": "base"}]
    data["relations"][1] = {
        "kind": "clutter_on",
        "subject": "cube",
        "reference": "surface",
        "params": {"clearance_m": 0.2, "random_yaw": False},
    }
    data["relations"].append({"kind": "is_anchor", "subject": "surface"})
    data["external_yaml"] = "physics.yaml"
    (tmp_path / "physics.yaml").write_text(
        yaml.safe_dump({
            "default_physics_backend": preset,
            "env_cfg_override": {"sim": {"dt": 0.01}, "decimation": 2},
        })
    )
    source.write_text(yaml.safe_dump(data))
    original = source.read_bytes()
    num_envs = 12 if preset == "newton" else 2
    min_layouts = num_envs + 1
    run_subprocess(
        [
            TestConstants.python_path,
            "-c",
            (
                "from isaaclab_arena.tests.test_settled_placement import run_cli_with_test_assets;"
                " run_cli_with_test_assets()"
            ),
            f"env_spec={source}",
            f"output={output}",
            f"num_envs={num_envs}",
            f"min_layouts={min_layouts}",
            "layouts_per_env=1",
            "max_batches=15",
            "settle.num_steps=120",
            "+settle.validators.support_containment.fall_through_tolerance_m=0.005",
            "--viz",
            "none",
        ],
        timeout_sec=180,
    )
    assert source.read_bytes() == original
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == min_layouts
    assert len({record["layout_id"] for record in records}) == min_layouts
    for record in records:
        assert record["source"] == "settled"
        assert set(record["poses"]) == {"cube_body", "table", "floor"}
        x, y, z = record["poses"]["cube_body"]["position_xyz"]
        assert -0.4 < x < 0.4 and -0.4 < y < 0.4
        assert z == pytest.approx(0.57, abs=0.005)
        reports = {report["check"]: report for report in record["validation"]["post_physics"]}
        assert reports["physics_settled"]["passed"]
        assert reports["support_containment"]["passed"]
        assert reports["support_containment"]["configuration"]["fall_through_tolerance_m"] == 0.005
        assert record["validation"]["sampling"]["physics_dt_s"] == pytest.approx(0.01)
        assert record["validation"]["sampling"]["decimation"] == 2


@pytest.mark.with_subprocess
def test_cli_generates_maintained_clutter(tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml"
    )
    output = tmp_path / "hammers.jsonl"
    run_subprocess(
        [
            TestConstants.python_path,
            str(SCRIPT),
            f"env_spec={source}",
            f"output={output}",
            "num_envs=1",
            "min_layouts=1",
            "layouts_per_env=1",
            "max_batches=15",
            "settle.num_steps=480",
            # The real table is beveled; its top is 0.530645 m in scaled local coordinates.
            "+settle.validators.support_containment.minimum_resting_heights_m.office_table_background=0.5306",
            "--viz",
            "none",
        ],
        timeout_sec=180,
    )
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert {"robot", "wood_hammer", "red_hammer", "blue_hammer", "clamp"} <= (record["poses"].keys())
    reports = {report["check"]: report for report in record["validation"]["post_physics"]}
    assert reports["physics_settled"]["passed"]
    assert reports["support_containment"]["passed"]
    assert reports["support_containment"]["configuration"]["minimum_resting_heights_m"] == {
        "office_table_background": 0.5306
    }
