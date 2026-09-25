# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks for the generated reference-backed RoboLab exact catalog."""

import json
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.assets.robolab_scene_factory import ROBOLAB_EXACT_SCENE_NAMES
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
REGULAR_DIR = REPO_ROOT / "isaaclab_arena_environments" / "robolab"
EXACT_DIR = REPO_ROOT / "isaaclab_arena_environments" / "robolab_exact"
ROBOLAB_METADATA = REPO_ROOT / "RoboLab" / "assets" / "scenes" / "_metadata" / "scene_metadata.json"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_robolab_exact_catalog_matches_regular_catalog():
    regular_scenes = sorted(path.name for path in (REGULAR_DIR / "scenes").glob("*.yaml"))
    exact_scenes = sorted(path.name for path in (EXACT_DIR / "scenes").glob("*.yaml"))
    regular_tasks = sorted(path.name for path in (REGULAR_DIR / "tasks").glob("*.yaml"))
    exact_tasks = sorted(path.name for path in (EXACT_DIR / "tasks").glob("*.yaml"))

    assert exact_scenes == regular_scenes
    assert exact_tasks == regular_tasks
    assert len(exact_scenes) == 17
    assert len(exact_tasks) == 38
    assert tuple(path.removesuffix(".yaml") for path in exact_scenes) == tuple(sorted(ROBOLAB_EXACT_SCENE_NAMES))


@pytest.mark.parametrize("task_path", sorted((EXACT_DIR / "tasks").glob("*.yaml")), ids=lambda path: path.stem)
def test_robolab_exact_task_specs_parse(task_path: Path):
    spec = ArenaEnvGraphSpec.from_yaml(task_path)
    scene_path = EXACT_DIR / "scenes" / Path(_load_yaml(task_path)["external_yaml"]).name
    scene = _load_yaml(scene_path)

    assert not spec.objects
    assert spec.object_references
    assert not spec.relations
    assert not scene["objects"]
    assert not scene["relations"]
    assert all(ref.object_type.value == "rigid" for ref in spec.object_references)
    assert all(
        subtask.params.get("background_scene", spec.background.id) == spec.background.id
        for subtask in spec.task.subtasks
    )


def test_robolab_exact_manifest_matches_source_metadata():
    if not ROBOLAB_METADATA.exists():
        pytest.skip("RoboLab source assets are not available")

    manifest = _load_yaml(EXACT_DIR / "SOURCE_MANIFEST.yaml")
    metadata = json.loads(ROBOLAB_METADATA.read_text(encoding="utf-8"))
    assert set(manifest["scenes"]) == set(ROBOLAB_EXACT_SCENE_NAMES)

    for scene_name, scene in manifest["scenes"].items():
        source_name = Path(scene["source_usda"]).name
        assert source_name == f"{scene_name}.usda"
        records = {record["prim_path"]: record for record in metadata[source_name]}
        for mapping in scene["object_mappings"]:
            record = records[mapping["prim_path"]]
            assert record["rigid_body"] is True
            assert mapping["object_type"] == "rigid"
