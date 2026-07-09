# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _test_kitchen_graph_attaches_embodiment_spatial_relations(simulation_app):
    from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "kitchen_embodiment_placement_env_graph.yaml")
    arena_env = build_arena_env_from_graph_spec(spec)

    spatial_relations = arena_env.embodiment.get_spatial_relations()
    assert len(spatial_relations) == 2
    relation_kinds = {type(relation).__name__ for relation in spatial_relations}
    assert relation_kinds == {"On", "NextTo"}
    return True


def test_kitchen_graph_attaches_embodiment_spatial_relations():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_kitchen_graph_attaches_embodiment_spatial_relations)


def _test_embodiment_rotate_around_solution_marker_is_attached(simulation_app):
    from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.relations import RotateAroundSolution

    data = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "kitchen_embodiment_placement_env_graph.yaml").to_dict()
    data["relations"].append({"kind": "rotate_around_solution", "subject": "droid", "params": {"yaw_rad": 1.57}})
    spec = ArenaEnvGraphSpec.from_dict(data)
    arena_env = build_arena_env_from_graph_spec(spec)

    markers = [rel for rel in arena_env.embodiment.get_relations() if isinstance(rel, RotateAroundSolution)]
    assert len(markers) == 1 and markers[0].yaw_rad == 1.57
    # The marker is not a spatial constraint, so it must not appear among the spatial relations.
    assert not any(isinstance(rel, RotateAroundSolution) for rel in arena_env.embodiment.get_spatial_relations())
    return True


def test_embodiment_rotate_around_solution_marker_is_attached():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_embodiment_rotate_around_solution_marker_is_attached)


def _test_embodiment_is_anchor_relation_is_rejected(simulation_app):
    from isaaclab_arena.environments.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

    data = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "kitchen_embodiment_placement_env_graph.yaml").to_dict()
    data["relations"].insert(0, {"kind": "is_anchor", "subject": "droid", "params": {}})
    spec = ArenaEnvGraphSpec.from_dict(data)

    with pytest.raises(AssertionError, match="Embodiment cannot be marked is_anchor"):
        build_arena_env_from_graph_spec(spec)
    return True


def test_embodiment_is_anchor_relation_is_rejected():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_embodiment_is_anchor_relation_is_rejected)
