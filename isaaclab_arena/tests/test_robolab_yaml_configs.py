# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static checks on the RoboLab scene, task and experiment-config YAMLs (no simulation app)."""

import yaml
from pathlib import Path

# RoboLab's PhysX solver limits, shared by every task.
ROBOLAB_PHYSICS = {"max_position_iteration_count": 32, "max_velocity_iteration_count": 1}

ROBOLAB_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "robolab"
SCENE_YAMLS = sorted((ROBOLAB_DIR / "scenes").glob("*.yaml"))


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test_every_robolab_scene_references_declared_nodes():
    assert SCENE_YAMLS, "no RoboLab scene YAMLs found"
    for scene_path in SCENE_YAMLS:
        scene = _load(scene_path)
        objects = scene["objects"]
        # Every relation must refer to a declared node (catches id typos and stale table references).
        node_ids = {scene["background"]["id"]} | {obj["id"] for obj in objects}
        for relation in scene.get("relations", []):
            assert relation["subject"] in node_ids, f"{scene_path.name}: unknown subject {relation['subject']}"
            reference = relation.get("reference")
            assert reference is None or reference in node_ids, f"{scene_path.name}: unknown reference {reference}"


TASK_YAMLS = sorted((ROBOLAB_DIR / "tasks").glob("*.yaml"))


def test_robolab_tasks_declare_lights():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    for task_path in TASK_YAMLS:
        # Loaded through the spec so lights missing from the shared defaults include are caught.
        spec = ArenaEnvGraphSpec.from_yaml(task_path)
        registry_names_by_id = {obj.id: obj.registry_name for obj in spec.objects}
        for light_id in ["light", "sphere_light"]:
            assert registry_names_by_id.get(light_id) == light_id, f"{task_path.name}: missing or wrong '{light_id}'"
        node_ids = {spec.background.id} | set(registry_names_by_id)
        assert len(node_ids) == len(spec.objects) + 1, f"{task_path.name}: duplicate node ids"


def test_robolab_tasks_have_solver_clamp():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    assert len(TASK_YAMLS) == 38
    for task_path in TASK_YAMLS:
        # Loaded through the spec so a clamp missing from the shared defaults include is caught.
        spec = ArenaEnvGraphSpec.from_yaml(task_path)
        expected = {"sim": {"physics": ROBOLAB_PHYSICS}}
        assert spec.env_cfg_override == expected, f"{task_path.name}: missing or wrong env_cfg_override"


def test_robolab_experiment_configs_leave_lighting_to_the_scene():
    config_paths = sorted((ROBOLAB_DIR / "experiment_configs").glob("robolab_*_tasks_pi0*.yaml"))
    assert len(config_paths) == 5
    for config_path in config_paths:
        variations = _load(config_path)["shared"].get("variations", {})
        for node_id in ["light", "sphere_light"]:
            assert node_id not in variations, f"{config_path.name}: sets variations on '{node_id}'"
