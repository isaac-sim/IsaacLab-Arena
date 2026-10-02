# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static authoring contracts and CLI checks, without SimulationApp or asset construction."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

import pytest

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.agentic_environment_generation.authoring_metadata import (
    AuthoringMetadata,
    ParameterMetadata,
    constructor_parameters,
)
from isaaclab_arena.agentic_environment_generation.catalogues import (
    AssetCatalogue,
    RelationCatalogue,
    build_asset_catalogue,
    build_catalogue_dict,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.semantic_validation import (
    collect_semantic_validation_issues,
    validate_authoring_spec,
)
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.tests.utils.agentic_environment_generation import minimal_spec_dict


class _Mode(str, Enum):
    SOFT = "soft"
    FIRM = "firm"


class _Graspable(AffordanceBase):
    pass


class _Object(Asset, _Graspable):
    tags = ["object"]
    object_type = ObjectType.RIGID
    authoring_metadata = AuthoringMetadata(provides=("sterile",))

    def __init__(self, label: Literal["left", "right"] = "left"):
        raise AssertionError("Discovery must not instantiate assets")


class _Task:
    agent_ready = True
    authoring_metadata = AuthoringMetadata(
        parameters={"distance": ParameterMetadata(units="m", minimum=0, maximum=1)},
        requires={"target": ("sterile",)},
        constraints=("Destination must be reachable.",),
        reset_semantics="Restores the initial object state on reset.",
    )

    def __init__(
        self,
        target: _Graspable,
        distance: float = 0.1,
        mode: _Mode = _Mode.SOFT,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0, 0, 0), (1, 1, 1)),
        optional: Asset | None = None,
    ):
        raise AssertionError("Discovery must not instantiate tasks")


class _Registry:
    def __init__(self, entries):
        self.entries = entries

    def get_all_keys(self):
        return list(self.entries)

    def get_asset_by_name(self, name):
        return self.entries[name]

    def get_task_by_name(self, name):
        return self.entries[name]


def test_catalogues_reflect_types_defaults_and_affordances_without_constructing():
    assets = build_asset_catalogue(_Registry({"tool": _Object})).to_dict()
    entry = assets["objects"][0]
    assert entry["parameters"]["label"] == {"enum": ["left", "right"], "required": False, "default": "left"}
    assert entry["provides"] == ["_Graspable", "rigid", "root_frame", "sterile"]
    task = build_task_catalogue(_Registry({"Task": _Task})).to_dict()["tasks"][0]
    assert task["parameters"]["distance"] == {
        "type": "number",
        "required": False,
        "default": 0.1,
        "x-units": "m",
        "minimum": 0,
        "maximum": 1,
    }
    assert task["parameters"]["mode"]["default"] == "soft"
    assert task["parameters"]["bounds"]["prefixItems"][0]["minItems"] == 3
    assert task["requires"] == {"target": ["_Graspable", "sterile"]}
    assert task["reset_semantics"] == _Task.authoring_metadata.reset_semantics
    json.dumps({"assets": assets, "task": task}, allow_nan=False)


def test_asset_factory_signature_is_discovered_without_calling(monkeypatch):
    from isaaclab_arena.assets.register import register_asset_factory
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    registry.get_all_keys()
    monkeypatch.setattr(registry, "_components", dict(registry._components))

    @register_asset_factory(name="authoring_test_factory", object_type=ObjectType.RIGID)
    def factory(component: Literal["block", "tray"] = "block", scale: float = 1.0):
        raise AssertionError("Discovery must not call asset factories")

    factory.authoring_metadata = AuthoringMetadata(parameters={"scale": ParameterMetadata(minimum=0.1)})
    entry = build_asset_catalogue(_Registry({factory.name: registry.get_asset_by_name(factory.name)})).objects[0]
    assert entry["parameters"]["component"]["enum"] == ["block", "tray"]
    assert entry["parameters"]["scale"]["minimum"] == 0.1


def _semantic_input(params):
    return {
        "embodiment": {"id": "robot", "registry_name": "robot"},
        "background": {"id": "table", "registry_name": "table"},
        "objects": [{"id": "good", "registry_name": "tool"}, {"id": "bad", "registry_name": "plain"}],
        "task": {"subtasks": [{"kind": "Task", "params": params}]},
    }


def _semantic_issues(params):
    assets = AssetCatalogue(
        embodiments=[{"name": "robot"}],
        backgrounds=[{"name": "table"}],
        objects=[{"name": "tool", "provides": ["_Graspable", "sterile"]}, {"name": "plain"}],
    )
    return collect_semantic_validation_issues(
        _semantic_input(params), assets, build_task_catalogue(_Registry({"Task": _Task})), RelationCatalogue()
    )


def test_semantic_validation_reports_capabilities_and_compatible_node_choices():
    issue = _semantic_issues({"target": "bad"})[0]
    assert issue.code == "missing_capability"
    assert issue.path == "/task/subtasks/0/params/target"
    assert issue.expected == {"capabilities": ["_Graspable", "sterile"]}
    assert issue.compatible_choices == ["good"]
    assert _semantic_issues({"target": "good", "mode": "soft", "optional": None}) == []


@pytest.mark.parametrize(
    "params,code,path",
    [
        ({"target": "missing"}, "unknown_node", "target"),
        ({}, "missing_parameter", "target"),
        ({"target": "good", "distance": -1}, "parameter_range", "distance"),
        ({"target": "good", "distance": True}, "parameter_type", "distance"),
        ({"target": "good", "distance": float("nan")}, "parameter_range", "distance"),
        ({"target": "good", "mode": "invalid"}, "parameter_enum", "mode"),
        ({"target": "good", "bounds": [[0, 0], [1, 1, 1]]}, "parameter_shape", "bounds/0"),
        ({"target": "good", "extra": 1}, "unsupported_parameter", "extra"),
    ],
)
def test_semantic_parameter_errors_have_exact_paths(params, code, path):
    assert any(
        issue.code == code and issue.path == f"/task/subtasks/0/params/{path}" for issue in _semantic_issues(params)
    )


def test_builtin_authoring_discovery_and_validation_are_static(monkeypatch):
    # Use actual registry entries and a shipped generic graph, without constructing assets.
    def fail(*args, **kwargs):
        raise AssertionError("Static authoring must not construct assets or environments")

    monkeypatch.setattr(Asset, "__init__", fail)
    monkeypatch.setattr(ArenaEnvGraphSpec, "to_arena_env", fail)
    catalogue = build_catalogue_dict()
    tasks = {entry["name"]: entry for entry in catalogue["tasks"]}
    assert "PickAndPlaceTask" in tasks
    assert "LiftObjectTask" not in tasks
    assert tasks["OpenDoorTask"]["requires"]["openable_object"] == ["Openable"]
    assert validate_authoring_spec(minimal_spec_dict())["valid"]
    data = minimal_spec_dict()
    data["task"]["subtasks"][0]["params"]["episode_length_s"] = "tomorrow"
    report = validate_authoring_spec(data)
    assert not report["valid"]
    assert report["issues"][0]["path"] == "/task/subtasks/0/params/episode_length_s"


def test_graph_node_validation_allows_literal_strings_and_checks_reference_collections(monkeypatch):
    from isaaclab_arena.assets.registries import TaskRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_task_conversion_utils import _resolve_node_refs_in_task_args

    class Task:
        def __init__(self, targets: list[Asset], label: str, optional: Asset | None = None):
            pass

    monkeypatch.setattr(TaskRegistry(), "get_task_by_name", lambda name: Task)
    data = minimal_spec_dict()
    target = data["objects"][0]["id"]
    data["task"]["subtasks"][0]["params"] = {"targets": [target], "label": "literal", "optional": None}
    spec = ArenaEnvGraphSpec.model_validate(data)
    obj = object()
    assert _resolve_node_refs_in_task_args(Task, spec.task.subtasks[0].params, {target: obj}) == {
        "targets": [obj],
        "label": "literal",
        "optional": None,
    }
    data["task"]["subtasks"][0]["params"]["targets"] = ["missing"]
    with pytest.raises(ValueError, match="references unknown node"):
        ArenaEnvGraphSpec.model_validate(data)


def test_cli_json_catalog_and_validation_outputs_parse_without_simulation(monkeypatch, capsys, tmp_path):
    from isaaclab_arena_examples.agentic_environment_generation import cli_runner

    def fail(*args, **kwargs):
        raise AssertionError("Discovery modes must not start SimulationApp")

    monkeypatch.setattr(cli_runner.SimulationAppContext, "__enter__", fail)
    monkeypatch.setattr("sys.argv", ["runner", "--mode", "catalog", "--format", "json"])
    assert cli_runner.main() == 0
    assert "tasks" in json.loads(capsys.readouterr().out)
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(minimal_spec_dict()))
    monkeypatch.setattr("sys.argv", ["runner", "--mode", "validate", "--format", "json", "--env_spec", str(path)])
    assert cli_runner.main() == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    path.write_text("objects: [")
    assert cli_runner.main() == 1
    assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == "input_error"


def test_unresolved_annotation_retains_other_parameter_metadata():
    class Task:
        def __init__(self, missing: UnavailableType, count: int = 3):  # noqa: F821
            pass

    parameters = constructor_parameters(Task)
    assert parameters["missing"]["x-python-type"] == "UnavailableType"
    assert parameters["count"]["type"] == "integer"


def test_generation_prompt_exposes_structured_contracts():
    from isaaclab_arena.agentic_environment_generation.spec_inference import SpecInference

    tasks = build_task_catalogue(_Registry({"Task": _Task}))
    message = SpecInference._user_message("place the tool", AssetCatalogue(), RelationCatalogue(), tasks)
    metadata = json.loads(message.split("AUTHORING METADATA:\n")[1].split("\n\nUSER PROMPT:")[0])
    assert metadata["tasks"][0]["requires"]["target"] == ["_Graspable", "sterile"]
    assert metadata["tasks"][0]["parameters"]["distance"]["x-units"] == "m"


def test_region_placement_checks_reference_types_and_all_object_set_members():
    from isaaclab_arena.tasks.place_in_region_task import PlaceInRegionTask

    assets = AssetCatalogue(
        objects=[{"name": "part", "object_type": "rigid"}, {"name": "fixture", "object_type": "base"}]
    )
    tasks = build_task_catalogue(_Registry({"PlaceInRegionTask": PlaceInRegionTask}))
    data = {
        "object_references": [
            {"id": "rigid_ref", "object_type": "rigid"},
            {"id": "base_ref", "object_type": "base"},
            {"id": "soft_ref", "object_type": "deformable"},
        ],
        "object_sets": [{"id": "mixed", "members": ["part", "fixture"]}, {"id": "parts", "members": ["part"]}],
        "task": {
            "subtasks": [{
                "kind": "PlaceInRegionTask",
                "params": {
                    "subject": "mixed",
                    "destination": "soft_ref",
                    "region_bounds": [[-1, -1, -1], [1, 1, 1]],
                },
            }]
        },
    }
    issues = collect_semantic_validation_issues(data, assets, tasks, RelationCatalogue())
    task_issues = [issue for issue in issues if issue.path.startswith("/task/")]
    assert len(task_issues) == 2
    assert task_issues[0].expected == {"capabilities": ["rigid"]}
    assert task_issues[0].compatible_choices == ["parts", "rigid_ref"]
    assert task_issues[1].expected == {"capabilities": ["root_frame"]}
    data["task"]["subtasks"][0]["params"].update(subject="rigid_ref", destination="base_ref")
    issues = collect_semantic_validation_issues(data, assets, tasks, RelationCatalogue())
    assert not any(issue.path.startswith("/task/") for issue in issues)
