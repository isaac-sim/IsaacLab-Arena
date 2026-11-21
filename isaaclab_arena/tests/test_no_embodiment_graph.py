# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Graph environments can omit the robot while retaining scene and task validation."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_graph_without_robot(_simulation_app):
    import argparse
    import yaml

    import pytest
    from pydantic import ValidationError

    from isaaclab_arena.agentic_environment_generation.catalogues import (
        build_asset_catalogue,
        build_relation_catalogue,
        build_task_catalogue,
    )
    from isaaclab_arena.agentic_environment_generation.spec_validation import collect_catalogue_validation_trace
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_examples.agentic_environment_generation.cli_runner import _iter_printable_assets
    from isaaclab_arena_examples.agentic_environment_generation.review_gui.spec_visualization.mermaid_graph import (
        render_mermaid_graph,
    )

    data = yaml.safe_load((Path(__file__).parent / "test_data/minimal_maple_table_env_graph.yaml").read_text())
    data.pop("embodiment")
    data["cli_override_specs"] = [{"arg": "object", "target_node_id": "rubiks_cube_hot3d_robolab"}]

    for embodiment_fields in ({}, {"embodiment": None}):
        spec = ArenaEnvGraphSpec.from_dict({**data, **embodiment_fields})
        assert spec.embodiment is None
        assert "embodiment" not in spec.to_dict()
        assert ArenaEnvGraphSpec.from_dict(spec.to_dict()).embodiment is None
        assert all(role != "embodiment" for role, *_ in _iter_printable_assets(spec))
        assert "franka_ik" not in render_mermaid_graph(spec)
        assert not collect_catalogue_validation_trace(
            spec, build_asset_catalogue(), build_task_catalogue(), build_relation_catalogue()
        )

        # Object overrides still work when no robot is available to override.
        spec.apply_cli_override_args(argparse.Namespace(object="dex_cube"))
        assert spec.objects[0].registry_name == "dex_cube"
        for enable_cameras in (False, True):
            arena_env = spec.to_arena_env(enable_cameras=enable_cameras)
            assert arena_env.embodiment is None
            assert arena_env.task.pick_up_object is arena_env.scene.assets["rubiks_cube_hot3d_robolab"]

    invalid_relation = {"kind": "on", "subject": "franka_ik", "reference": "maple_table_robolab"}
    with pytest.raises(ValidationError, match="unknown subject 'franka_ik'"):
        ArenaEnvGraphSpec.from_dict({**data, "relations": [*data["relations"], invalid_relation]})
    with pytest.raises(ValidationError, match="unknown or non-swappable asset 'franka_ik'"):
        ArenaEnvGraphSpec.from_dict({
            **data,
            "cli_override_specs": [{"arg": "robot", "target_node_id": "franka_ik"}],
        })

    return True


def test_graph_without_robot():
    assert run_function_with_persistent_simulation_app(_test_graph_without_robot)
