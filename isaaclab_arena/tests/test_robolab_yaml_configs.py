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


TASK_YAMLS = sorted((ROBOLAB_DIR / "tasks").glob("*.yaml"))

SOLVER_CLAMP = {"sim": {"physics": {"max_position_iteration_count": 32, "max_velocity_iteration_count": 1}}}

# RoboLab episode_length_s per task (from the RoboLab task definitions), keyed by task YAML stem.
ROBOLAB_EPISODE_LENGTH_S = {
    "apple_and_yogurt_in_bowl": 120.0,
    "bagels_on_plate": 60.0,
    "banana_in_bowl": 50.0,
    "banana_on_plate": 40.0,
    "banana_then_rubiks_cube": 60.0,
    "bbq_sauce_in_bin": 90.0,
    "big_pumpkin_in_bin": 60.0,
    "black_items_in_bin": 120.0,
    "bowl_in_bin": 60.0,
    "butter_above_raisin": 40.0,
    "canned_food_in_bin": 60.0,
    "clamp_in_right_bin": 60.0,
    "clear_organic_objects": 240.0,
    "clutter_plastic": 180.0,
    "clutter_pumpkin": 90.0,
    "coffee_pot_in_bin": 60.0,
    "condiments_in_bin": 180.0,
    "electronics_in_bin": 180.0,
    "food_packing_1_boxes": 60.0,
    "food_packing_1_cans": 60.0,
    "hammers_in_left_bin": 180.0,
    "larger_object_raisin_box_in_bin": 30.0,
    "marker_in_mug": 40.0,
    "mouse_on_keyboard": 60.0,
    "mustard_above_raisin": 40.0,
    "mustard_in_left_bin": 30.0,
    "mustard_in_right_bin": 30.0,
    "non_hammer_tools_in_right_bin": 180.0,
    "rubiks_cube": 40.0,
    "rubiks_cube_and_banana": 60.0,
    "rubiks_cube_then_banana": 60.0,
    "sauce_bottles_crate": 40.0,
    "small_pumpkin_in_bin": 60.0,
    "smaller_object_butter_in_bin": 30.0,
    "smartphone_in_bin": 60.0,
    "throw_away_apple": 60.0,
    "throw_away_snacks": 120.0,
    "toy_in_bin": 60.0,
}


def test_robolab_tasks_have_solver_clamp():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    assert len(TASK_YAMLS) == 38
    for task_path in TASK_YAMLS:
        # Loaded through the spec so a clamp dropped by the external_yaml scene merge is caught.
        spec = ArenaEnvGraphSpec.from_yaml(task_path)
        assert spec.env_cfg_override == SOLVER_CLAMP, f"{task_path.name}: missing or wrong env_cfg_override"


def test_robolab_task_episode_lengths_match_robolab():
    assert {path.stem for path in TASK_YAMLS} == set(ROBOLAB_EPISODE_LENGTH_S)
    for task_path in TASK_YAMLS:
        task = _load(task_path)["task"]
        total = sum(subtask["params"]["episode_length_s"] for subtask in task["subtasks"])
        expected = ROBOLAB_EPISODE_LENGTH_S[task_path.stem]
        assert abs(total - expected) < 1e-6, f"{task_path.name}: subtask lengths sum to {total}, RoboLab has {expected}"
        if expected != 70.0:
            for subtask in task["subtasks"]:
                assert subtask["params"]["episode_length_s"] != 70.0, f"{task_path.name}: still has the old 70.0 s"
