# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent-generated spec validation."""

from isaaclab_arena.agentic_environment_generation.catalogues import (
    AssetCatalogue,
    RelationCatalogue,
    RelationCatalogueEntry,
    TaskCatalogue,
)
from isaaclab_arena.agentic_environment_generation.spec_validation import collect_agent_ready_validation_trace
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import (
    catalog,
    minimal_spec_dict,
    relation_catalog,
    task_catalog,
)


def _minimal_spec() -> ArenaEnvGraphSpec:
    return ArenaEnvGraphSpec.model_validate(minimal_spec_dict())


def test_collect_agent_ready_validation_trace_accepts_catalogued_spec():
    traces = collect_agent_ready_validation_trace(
        _minimal_spec(),
        asset_catalog=catalog("ASSETS"),
        task_catalog=task_catalog("TASKS"),
        relation_catalog=relation_catalog("RELATIONS"),
    )

    assert traces == []


def test_collect_agent_ready_validation_trace_rejects_assets_outside_subcatalogs():
    traces = collect_agent_ready_validation_trace(
        _minimal_spec(),
        asset_catalog=AssetCatalogue(),
        task_catalog=task_catalog("TASKS"),
        relation_catalog=relation_catalog("RELATIONS"),
    )

    assert "Embodiment registry_name 'franka_ik' is not in the EMBODIMENTS catalog" in traces
    assert "Background registry_name 'maple_table_robolab' is not in the BACKGROUNDS catalog" in traces
    assert "Object 'rubiks_cube_hot3d_robolab' registry_name 'rubiks_cube_hot3d_robolab'" in " ".join(traces)
    assert "Object 'bowl_ycb_robolab' registry_name 'bowl_ycb_robolab'" in " ".join(traces)


def test_collect_agent_ready_validation_trace_rejects_unknown_missing_and_unsupported_task_params():
    spec = _minimal_spec()
    task = spec.task.subtasks[0]

    unknown_traces = collect_agent_ready_validation_trace(
        spec,
        asset_catalog=catalog("ASSETS"),
        task_catalog=TaskCatalogue(),
        relation_catalog=relation_catalog("RELATIONS"),
    )
    assert "Task 'PickAndPlaceTask' is not in the TASKS catalog" in unknown_traces

    del task.params["pick_up_object"]
    task.params["hallucinated_param"] = True
    param_traces = collect_agent_ready_validation_trace(
        spec,
        asset_catalog=catalog("ASSETS"),
        task_catalog=task_catalog("TASKS"),
        relation_catalog=relation_catalog("RELATIONS"),
    )
    assert "Task 'PickAndPlaceTask' is missing required param 'pick_up_object'" in param_traces
    assert any("Task 'PickAndPlaceTask' has unsupported param 'hallucinated_param'" in trace for trace in param_traces)


def test_collect_agent_ready_validation_trace_rejects_unknown_missing_and_unsupported_relation_params():
    spec = _minimal_spec()
    unknown_traces = collect_agent_ready_validation_trace(
        spec,
        asset_catalog=catalog("ASSETS"),
        task_catalog=task_catalog("TASKS"),
        relation_catalog=RelationCatalogue(),
    )
    assert "Relation 'is_anchor' is not in the RELATIONS catalog" in unknown_traces
    assert "Relation 'on' is not in the RELATIONS catalog" in unknown_traces

    spec.relations[1].params["hallucinated_param"] = True
    relations = RelationCatalogue(
        relations=[
            RelationCatalogueEntry("is_anchor", True, [], [], {}, ""),
            RelationCatalogueEntry("on", False, ["distance_m"], [], {}, ""),
        ]
    )
    param_traces = collect_agent_ready_validation_trace(
        spec,
        asset_catalog=catalog("ASSETS"),
        task_catalog=task_catalog("TASKS"),
        relation_catalog=relations,
    )
    assert "Relation 'on' is missing required param 'distance_m'" in param_traces
    assert any("Relation 'on' has unsupported param 'hallucinated_param'" in trace for trace in param_traces)
