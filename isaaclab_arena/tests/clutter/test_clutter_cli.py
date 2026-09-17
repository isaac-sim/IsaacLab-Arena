# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter command-line parsing and cache generation."""

import subprocess
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants

CLUTTER_DIR = Path(__file__).parents[3] / "isaaclab_arena_examples/relations/clutter"
SCRIPT = Path(TestConstants.scripts_dir) / "generate_clutter_scene.py"


def _arguments(output, source=CLUTTER_DIR / "clutter_scene.yaml"):
    return [
        "--env_spec",
        str(source),
        "--output",
        str(output),
    ]


def _run(arguments):
    return subprocess.run(
        [TestConstants.python_path, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        timeout=180,
    )


def register_no_embodiment():
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment

    AssetRegistry().register(NoEmbodiment, key="clutter_test_no_embodiment")


def test_help_lists_settling_controls():
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    assert "--passive_move_thresh_m" in result.stdout
    assert "--required_quiet_windows" in result.stdout


def test_settle_import_does_not_load_usd():
    result = subprocess.run(
        [
            TestConstants.python_path,
            "-c",
            (
                "import sys; import isaaclab_arena.relations.clutter.settle; "
                "assert 'pxr' not in sys.modules, 'USD imported before SimulationApp startup'"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--num_envs", "0"], "must be positive"),
        (["--attempts", "0"], "must be positive"),
        (["--register", "missing_colon"], "expected module:function"),
        (["--num_layouts", "0"], "must be positive"),
    ],
)
def test_invalid_options_fail_without_writing(tmp_path, arguments, message):
    output = tmp_path / "scene.yaml"
    result = _run([*_arguments(output), *arguments])
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


def test_missing_required_arguments():
    result = _run([])
    assert result.returncode != 0
    assert "required" in result.stderr


def test_existing_output_is_rejected(tmp_path):
    output = tmp_path / "scene.yaml"
    output.write_text("existing scene")
    result = _run(_arguments(output))
    assert result.returncode != 0
    assert "Output already exists" in result.stderr
    assert output.read_text() == "existing scene"


@pytest.mark.with_subprocess
@pytest.mark.parametrize("preset", [None, "newton"])
def test_cli_generates_scene_cache(tmp_path, preset):
    import yaml

    output = tmp_path / "scene.yaml"
    source = CLUTTER_DIR / "clutter_scene.yaml"
    if preset is not None:
        # The office table has authored inertia that MuJoCo rejects.
        support = tmp_path / "support.usda"
        support.write_text("""#usda 1.0
(
    defaultPrim = "Support"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Support" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    bool physics:kinematicEnabled = true
    float physics:mass = 1
    def Xform "surface" {
        double3 xformOp:translate = (0, 0, 0.7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        ) {
            double size = 1
            double3 xformOp:scale = (1, 1, 0.1)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }
    }
}
""")
        data = yaml.safe_load((CLUTTER_DIR / "clutter_scene.yaml").read_text())
        data["background"] = {
            "id": "table",
            "registry_name": "simready_usd_object",
            "params": {"usd_path": str(support), "instance_name": "table"},
        }
        data["embodiment"] = {"id": "robot", "registry_name": "clutter_test_no_embodiment"}
        data["object_references"] = [
            {"id": "surface", "parent_id": "table", "prim_path": "surface", "object_type": "base"}
        ]
        data["relations"].append({"kind": "is_anchor", "subject": "surface"})
        for relation in data["relations"]:
            if relation["kind"] == "clutter_on":
                relation["reference"] = "surface"
        source = tmp_path / "source.yaml"
        source.write_text(yaml.safe_dump(data))
    # Double-digit environment IDs catch lexicographic pose-row ordering.
    num_envs = 12 if preset == "newton" else 2
    num_layouts = num_envs + 1
    arguments = [
        *_arguments(output, source),
        "--num_envs",
        str(num_envs),
        "--num_layouts",
        str(num_layouts),
        "--viz",
        "none",
    ]
    if preset is not None:
        arguments.extend([
            "--presets",
            preset,
            "--register",
            "isaaclab_arena.tests.clutter.test_clutter_cli:register_no_embodiment",
        ])
    result = _run(arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    spec = yaml.safe_load(output.read_text())
    assert set(spec) == {f"cube_{i}" for i in range(4)}
    for poses in spec.values():
        assert len(poses) == num_layouts
        for pose in poses:
            assert len(pose["position_xyz"]) == 3
            assert len(pose["rotation_xyzw"]) == 4
