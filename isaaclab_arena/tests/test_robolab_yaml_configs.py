# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static checks on the RoboLab scene, task and experiment-config YAMLs (no simulation app)."""

import yaml
from pathlib import Path

ROBOLAB_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "robolab"
SCENE_YAMLS = sorted((ROBOLAB_DIR / "scenes").glob("*.yaml"))


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test_every_robolab_scene_declares_lights():
    assert SCENE_YAMLS, "no RoboLab scene YAMLs found"
    for scene_path in SCENE_YAMLS:
        scene = _load(scene_path)
        objects = scene["objects"]
        for light_id in ["light", "sphere_light"]:
            matches = [obj for obj in objects if obj["id"] == light_id]
            assert len(matches) == 1, f"{scene_path.name}: expected exactly one object with id '{light_id}'"
            assert matches[0]["registry_name"] == light_id, f"{scene_path.name}: '{light_id}' has wrong registry_name"

        # Every relation must refer to a declared node (catches id typos and stale table references).
        node_ids = {scene["background"]["id"]} | {obj["id"] for obj in objects}
        for relation in scene.get("relations", []):
            assert relation["subject"] in node_ids, f"{scene_path.name}: unknown subject {relation['subject']}"
            reference = relation.get("reference")
            assert reference is None or reference in node_ids, f"{scene_path.name}: unknown reference {reference}"
