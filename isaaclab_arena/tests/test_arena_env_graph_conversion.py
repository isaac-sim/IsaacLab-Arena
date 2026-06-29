# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for graph-spec -> live IsaacLabArenaEnvironment conversion.

Lives apart from ``test_arena_env_graph_spec.py`` on purpose: that file's in-process tests call
``spec.validate()``, which transitively imports ``pxr`` (relation-class resolution). The
persistent in-process ``SimulationApp`` here cannot start if ``pxr`` was imported first, so the
sim test must not share a process with those pxr-importing tests. Keeping it solo lets the app
launch cleanly before any ``pxr`` import.
"""

from pathlib import Path

import pytest

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _test_arena_env_graph_conversion_builds_sequential_pick_and_place_task(simulation_app):
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")
    arena_env = spec.to_arena_env()

    assert arena_env.name == "pick_and_place_maple_table_default"
    assert isinstance(arena_env.task, SequentialTaskBase)
    assert arena_env.task.desired_subtask_success_state == [True, True]
    assert len(arena_env.task.subtasks) == 2
    assert all(isinstance(subtask, PickAndPlaceTask) for subtask in arena_env.task.subtasks)
    assert arena_env.task.subtasks[0].pick_up_object.name == "rubiks_cube_hot3d_robolab"
    assert arena_env.task.subtasks[1].pick_up_object.name == "mug_ycb_robolab"
    assert all(subtask.destination_location.name == "bowl_ycb_robolab" for subtask in arena_env.task.subtasks)
    assert all(subtask.background_scene.name == "maple_table_robolab" for subtask in arena_env.task.subtasks)

    return True


def test_arena_env_graph_conversion_builds_sequential_pick_and_place_task():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    result = run_simulation_app_function(_test_arena_env_graph_conversion_builds_sequential_pick_and_place_task)
    assert result


def _test_get_arena_builder_from_cli_builds_env_from_graph_yaml(simulation_app):
    import argparse
    import sys

    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    yaml_path = str(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")

    # --env_graph_spec_yaml with no example-environment subcommand: parses (subcommand is
    # optional) and the runner builds the env from the graph spec instead of the registry.
    sys.argv = ["policy_runner.py", "--env_graph_spec_yaml", yaml_path]
    args = get_isaaclab_arena_environments_cli_parser().parse_args()

    builder = get_arena_builder_from_cli(args)
    assert builder.arena_env.name == "pick_and_place_maple_table_default"

    # The flags the YAML declares under `cli_override_specs` are registered dynamically by the
    # environments parser (not hardcoded). Confirm --object parses through that real parser
    # path and that apply_cli_override_args swaps the declared target node's asset name.
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

    sys.argv = ["policy_runner.py", "--env_graph_spec_yaml", yaml_path, "--object", "dex_cube"]
    args = get_isaaclab_arena_environments_cli_parser().parse_args()
    assert args.object == "dex_cube"
    spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
    spec.apply_cli_override_args(args)
    assert spec.nodes_by_id["rubiks_cube_hot3d_robolab"].name == "dex_cube"

    # A non-existent --env_graph_spec_yaml fails with a clear "not found" assertion from the YAML
    # loader, not an opaque FileNotFoundError. The parser hits it while building, when it reads the
    # graph's declared override flags.
    sys.argv = ["policy_runner.py", "--env_graph_spec_yaml", "/no/such/env_graph.yaml"]
    with pytest.raises(AssertionError, match="not found"):
        get_isaaclab_arena_environments_cli_parser()

    # Neither source, or both at once, is rejected by the exactly-one-source assert.
    for bad in (
        argparse.Namespace(env_graph_spec_yaml=None, example_environment=None),
        argparse.Namespace(env_graph_spec_yaml=yaml_path, example_environment="lift_object"),
    ):
        with pytest.raises(AssertionError):
            get_arena_builder_from_cli(bad)

    return True


def test_get_arena_builder_from_cli_builds_env_from_graph_yaml():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    result = run_simulation_app_function(_test_get_arena_builder_from_cli_builds_env_from_graph_yaml)
    assert result


def _test_default_light_is_injected_when_scene_has_none(simulation_app):
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeSpec, ArenaEnvGraphNodeType

    # A single YCB object with no light node and no light baked into its USD: the converter
    # must inject a default light so the env does not render black.
    spec = ArenaEnvGraphSpec(
        env_name="lighting_injection_test",
        nodes=[ArenaEnvGraphNodeSpec(id="mug", name="mug_ycb_robolab", type=ArenaEnvGraphNodeType.OBJECT)],
    )
    arena_env = spec.to_arena_env()

    # The injected light is tracked as a real LIGHTING node (so the augmented graph stays faithful)
    # and materializes as a DomeLight scene asset.
    light_nodes = [node for node in spec.nodes if node.type == ArenaEnvGraphNodeType.LIGHTING]
    assert len(light_nodes) == 1, f"expected one injected light node, got {light_nodes}"
    assert any(isinstance(asset, DomeLight) for asset in arena_env.scene.assets.values())

    # An explicit light suppresses injection — no double-lighting.
    explicit = ArenaEnvGraphSpec(
        env_name="lighting_explicit_test",
        nodes=[
            ArenaEnvGraphNodeSpec(id="mug", name="mug_ycb_robolab", type=ArenaEnvGraphNodeType.OBJECT),
            ArenaEnvGraphNodeSpec(id="my_light", name="light", type=ArenaEnvGraphNodeType.LIGHTING),
        ],
    )
    num_nodes_before = len(explicit.nodes)
    explicit_env = explicit.to_arena_env()
    assert len(explicit.nodes) == num_nodes_before, "must not inject a light when one is already declared"
    assert sum(isinstance(asset, DomeLight) for asset in explicit_env.scene.assets.values()) == 1

    return True


def test_default_light_is_injected_when_scene_has_none():
    pytest.importorskip("isaaclab.app")

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    result = run_simulation_app_function(_test_default_light_is_injected_when_scene_has_none)
    assert result
