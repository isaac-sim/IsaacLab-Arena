# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the workbench cleanup scene and tasks."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_ENVIRONMENT_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/workbench_cleanup"


def _test_workbench_task_definitions_build(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

    for path in sorted(_ENVIRONMENT_DIR.glob("task_*.yaml")):
        environment = build_arena_env_from_graph_spec(ArenaEnvGraphSpec.from_yaml(path))
        assert environment.name
    return True


def test_workbench_task_definitions_build():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_task_definitions_build)


def _test_joint_triggered_material_toggle(simulation_app):
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg
    from pxr import Usd, UsdGeom, UsdShade

    from isaaclab_arena.terms.events import JointTriggeredMaterialToggle

    stage = Usd.Stage.CreateInMemory()
    roots = [f"/World/envs/env_{index}/device" for index in range(2)]
    for root in roots:
        UsdGeom.Subset.Define(stage, f"{root}/Visuals/Indicator")
        UsdShade.Material.Define(stage, f"{root}/Materials/Default")
        UsdShade.Material.Define(stage, f"{root}/Materials/Alternate")

    class Scene(dict):
        env_prim_paths = [root.rsplit("/", 1)[0] for root in roots]
        env_regex_ns = "/World/envs/env_[^/]+"

    q = torch.zeros(2, 1)
    asset = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/device"),
        data=SimpleNamespace(
            joint_names=["power"],
            joint_pos=SimpleNamespace(torch=q),
            joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-0.00235, 0]]] * 2)),
        ),
    )
    env = SimpleNamespace(
        scene=Scene(device=asset),
        sim=SimpleNamespace(is_rendering=False),
        num_envs=2,
        device="cpu",
    )
    params = {
        "asset_name": "device",
        "joint_name": "power",
        "target_prim_path": "Visuals/Indicator",
        "default_material_path": "Materials/Default",
        "alternate_material_path": "Materials/Alternate",
    }
    with sim_utils.use_stage(stage):
        event = JointTriggeredMaterialToggle(
            EventTermCfg(func=JointTriggeredMaterialToggle, mode="interval", params=params), env
        )
    q[0] = -0.00235
    event(env, None, **params)
    assert event.alternate_active.tolist() == [True, False]
    event(env, None, **params)
    assert event.alternate_active.tolist() == [True, False]
    q[0] = 0
    event(env, None, **params)
    q[:] = -0.00235
    event(env, None, **params)
    assert event.alternate_active.tolist() == [False, True]
    event.reset(torch.tensor([0]))
    assert event.alternate_active.tolist() == [False, True]
    event.reset(slice(None))
    assert event.alternate_active.tolist() == [False, False]
    return True


def test_joint_triggered_material_toggle():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_joint_triggered_material_toggle)


@pytest.mark.parametrize(
    "filename,task_kind",
    [
        ("task_bin_shelf.yaml", "PickAndPlaceTask"),
        ("task_tool_in_drawer.yaml", "PickAndPlaceTask"),
        ("task_turn_off_scope.yaml", "PressButtonTask"),
        ("task_close_toolcase.yaml", "CloseDoorTask"),
        ("task_close_drawer.yaml", "CloseDoorTask"),
    ],
)
def test_workbench_task_uses_shared_scene(filename, task_kind):
    spec = ArenaEnvGraphSpec.from_yaml(_ENVIRONMENT_DIR / filename)

    assert spec.background.params["usd_path"] == "/datasets/workbench/assets/workbench.usdc"
    assert spec.task.subtasks[0].kind == task_kind
    assert spec.events["oscilloscope_power"].mode == "interval"
    socket_set = next(obj for obj in spec.objects if obj.id == "socket_set")
    assert socket_set.registry_name == "simready_usd_openable_articulation"
    assert socket_set.params["usd_path"] == "/datasets/workbench/assets/socket_set_open.usdc"
    object_ids = {obj.id for obj in spec.objects}
    assert "slotted_screwdriver" in object_ids
    assert "small_wrench" not in object_ids
    tool_on_table = next(
        relation for relation in spec.relations if relation.kind == "on" and relation.subject == "slotted_screwdriver"
    )
    assert tool_on_table.reference == "tabletop"
    references = {reference.id: reference for reference in spec.object_references}
    assert references["socket_set_lid"].parent_id == "socket_set"
    assert references["top_drawer_body"].parent_id == "top_drawer"
    assert references["top_drawer"].params["initial_joint_positions"] == {"joint_drawersA_01_joint_00": 0.251816005}


def test_workbench_full_cleanup_is_sequential():
    spec = ArenaEnvGraphSpec.from_yaml(_ENVIRONMENT_DIR / "task_full_cleanup.yaml")

    assert spec.task.composition.value == "sequential"
    assert [subtask.kind for subtask in spec.task.subtasks] == [
        "PickAndPlaceTask",
        "CloseDoorTask",
        "CloseDoorTask",
        "PickAndPlaceTask",
        "PressButtonTask",
    ]
