# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static checks on the RoboLab scene, task and experiment-config YAMLs (no simulation app)."""

import yaml
from pathlib import Path

# RoboLab's per-task settings (episode length from the RoboLab task definitions, PhysX solver limits, light rig),
# keyed by task YAML stem. Every task currently shares the same physics and light settings.
ROBOLAB_PHYSICS = {"max_position_iteration_count": 32, "max_velocity_iteration_count": 1}
ROBOLAB_LIGHT = {
    "hdr_names": ["home_office_robolab"],
    "dome_intensity": [500.0],
    "dome_color": [1.0, 1.0, 1.0],
    "sphere_intensity": [5000.0],
}


def _constants(episode_length_s: float) -> dict:
    return {"episode_length_s": episode_length_s, "physics": ROBOLAB_PHYSICS, "light": ROBOLAB_LIGHT}


ROBOLAB_TASK_CONSTANTS = {
    "apple_and_yogurt_in_bowl": _constants(120.0),
    "bagels_on_plate": _constants(60.0),
    "banana_in_bowl": _constants(50.0),
    "banana_on_plate": _constants(40.0),
    "banana_then_rubiks_cube": _constants(60.0),
    "bbq_sauce_in_bin": _constants(90.0),
    "big_pumpkin_in_bin": _constants(60.0),
    "black_items_in_bin": _constants(120.0),
    "bowl_in_bin": _constants(60.0),
    "butter_above_raisin": _constants(40.0),
    "canned_food_in_bin": _constants(60.0),
    "clamp_in_right_bin": _constants(60.0),
    "clear_organic_objects": _constants(240.0),
    "clutter_plastic": _constants(180.0),
    "clutter_pumpkin": _constants(90.0),
    "coffee_pot_in_bin": _constants(60.0),
    "condiments_in_bin": _constants(180.0),
    "electronics_in_bin": _constants(180.0),
    "food_packing_1_boxes": _constants(60.0),
    "food_packing_1_cans": _constants(60.0),
    "hammers_in_left_bin": _constants(180.0),
    "larger_object_raisin_box_in_bin": _constants(30.0),
    "marker_in_mug": _constants(40.0),
    "mouse_on_keyboard": _constants(60.0),
    "mustard_above_raisin": _constants(40.0),
    "mustard_in_left_bin": _constants(30.0),
    "mustard_in_right_bin": _constants(30.0),
    "non_hammer_tools_in_right_bin": _constants(180.0),
    "rubiks_cube": _constants(40.0),
    "rubiks_cube_and_banana": _constants(60.0),
    "rubiks_cube_then_banana": _constants(60.0),
    "sauce_bottles_crate": _constants(40.0),
    "small_pumpkin_in_bin": _constants(60.0),
    "smaller_object_butter_in_bin": _constants(30.0),
    "smartphone_in_bin": _constants(60.0),
    "throw_away_apple": _constants(60.0),
    "throw_away_snacks": _constants(120.0),
    "toy_in_bin": _constants(60.0),
}

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


def test_robolab_tasks_have_solver_clamp():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    assert len(TASK_YAMLS) == 38
    for task_path in TASK_YAMLS:
        # Loaded through the spec so a clamp dropped by the external_yaml scene merge is caught.
        spec = ArenaEnvGraphSpec.from_yaml(task_path)
        expected = {"sim": {"physics": ROBOLAB_TASK_CONSTANTS[task_path.stem]["physics"]}}
        assert spec.env_cfg_override == expected, f"{task_path.name}: missing or wrong env_cfg_override"


def test_robolab_task_episode_lengths_match_robolab():
    assert {path.stem for path in TASK_YAMLS} == set(ROBOLAB_TASK_CONSTANTS)
    for task_path in TASK_YAMLS:
        task = _load(task_path)["task"]
        total = sum(subtask["params"]["episode_length_s"] for subtask in task["subtasks"])
        expected = ROBOLAB_TASK_CONSTANTS[task_path.stem]["episode_length_s"]
        assert abs(total - expected) < 1e-6, f"{task_path.name}: subtask lengths sum to {total}, RoboLab has {expected}"
        if expected != 70.0:
            for subtask in task["subtasks"]:
                assert subtask["params"]["episode_length_s"] != 70.0, f"{task_path.name}: still has the old 70.0 s"


def test_robolab_experiment_configs_set_lighting():
    config_paths = sorted((ROBOLAB_DIR / "experiment_configs").glob("robolab_*_tasks_pi0*.yaml"))
    assert len(config_paths) == 5
    for config_path in config_paths:
        config = _load(config_path)
        variations = config["shared"]["variations"]
        for run_name, run in config["runs"].items():
            environment_path = ROBOLAB_DIR.parents[1] / run["environment"]["type"]
            assert environment_path.exists(), f"{config_path.name}: {run_name} environment {environment_path} missing"

            light = ROBOLAB_TASK_CONSTANTS[environment_path.stem]["light"]
            assert variations["light"]["hdr_image"]["hdr_names"] == light["hdr_names"], config_path.name
            expected_samplers = {
                ("light", "intensity"): light["dome_intensity"],
                ("light", "color"): light["dome_color"],
                ("sphere_light", "intensity"): light["sphere_intensity"],
            }
            for (node_id, variation_name), value in expected_samplers.items():
                variation = variations[node_id][variation_name]
                assert variation["enabled"] is True, f"{config_path.name}: {node_id}.{variation_name} not enabled"
                assert variation["sampler_cfg"]["low"] == value, f"{config_path.name}: {node_id}.{variation_name} low"
                assert variation["sampler_cfg"]["high"] == value, f"{config_path.name}: {node_id}.{variation_name} high"
