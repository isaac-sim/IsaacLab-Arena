# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discovery reuses existing registrations and validation without constructing assets."""

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
from isaaclab_arena.agentic_environment_generation.catalogues import build_asset_catalogue, build_task_catalogue
from isaaclab_arena.agentic_environment_generation.semantic_validation import validate_authoring_spec
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

    def __init__(self, label: Literal["left", "right"] = "left"):
        raise AssertionError("Discovery must not instantiate assets")


class _Task:
    agent_ready = True
    authoring_metadata = AuthoringMetadata(
        parameters={"distance": ParameterMetadata(units="m", minimum=0)},
        reset_semantics="Restores the initial object state on reset.",
    )

    def __init__(self, target: _Graspable, distance: float = 0.1, mode: _Mode = _Mode.SOFT):
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
    assert "_Graspable" in entry["provides"]
    task = build_task_catalogue(_Registry({"Task": _Task})).to_dict()["tasks"][0]
    assert task["parameters"]["distance"] == {
        "type": "number",
        "required": False,
        "default": 0.1,
        "x-units": "m",
        "minimum": 0,
    }
    assert task["parameters"]["mode"]["default"] == "soft"
    assert task["requires"] == {"target": ["_Graspable"]}
    assert task["reset_semantics"] == _Task.authoring_metadata.reset_semantics
    json.dumps({"assets": assets, "task": task}, allow_nan=False)


@pytest.mark.parametrize(
    "annotation",
    [_Mode, _Mode | None, list[_Mode], tuple[_Mode, int], dict[str, _Mode], Literal["soft", "firm"]],
)
def test_text_catalogue_preserves_choices_from_parameter_schemas(annotation):
    class Task:
        """Choose a mode."""

        agent_ready = True

        def __init__(self, choice, retries: int = 2, task_description: str = "ignored", **kwargs):
            raise AssertionError("Discovery must not construct tasks")

    Task.__init__.__annotations__["choice"] = annotation
    catalogue = build_task_catalogue(_Registry({"Task": Task}))
    assert "task_description" not in catalogue.tasks[0].parameters
    assert (
        catalogue.to_catalog_string()
        == "TASKS (1):\n- Task (required: choice={soft, firm}; optional: retries): Choose a mode."
    )


def test_unresolved_annotation_retains_other_parameter_metadata():
    class Task:
        def __init__(self, missing: UnavailableType, count: int = 3):  # noqa: F821
            pass

    parameters = constructor_parameters(Task)
    assert parameters["missing"]["x-python-type"] == "UnavailableType"
    assert parameters["count"]["type"] == "integer"


@pytest.mark.parametrize("error_kind", ("schema", "catalogue"))
def test_validation_reports_existing_schema_and_catalogue_errors(error_kind):
    data = minimal_spec_dict()
    if error_kind == "schema":
        del data["background"]
    else:
        data["task"]["subtasks"][0]["params"]["unknown_parameter"] = 1
    report = validate_authoring_spec(data)
    assert not report["valid"]
    assert report["validation_scope"] == "schema_and_catalogue"
    issue = report["issues"][0]
    assert issue["code"] == f"{error_kind}_error"
    if error_kind == "schema":
        assert issue["path"] == "/background"
    else:
        assert "unsupported param 'unknown_parameter'" in issue["message"]


def test_cli_discovery_outputs_parse_without_simulation_or_asset_construction(monkeypatch, capsys, tmp_path):
    from isaaclab_arena_examples.agentic_environment_generation import cli_runner

    def fail(*args, **kwargs):
        raise AssertionError("Discovery must not start simulation or construct assets")

    monkeypatch.setattr(cli_runner.SimulationAppContext, "__enter__", fail)
    monkeypatch.setattr(Asset, "__init__", fail)
    monkeypatch.setattr(ArenaEnvGraphSpec, "to_arena_env", fail)
    monkeypatch.setattr("sys.argv", ["runner", "--mode", "catalog", "--format", "json"])
    assert cli_runner.main() == 0
    catalogue = json.loads(capsys.readouterr().out)
    tasks = {entry["name"]: entry for entry in catalogue["tasks"]}
    assert tasks["OpenDoorTask"]["requires"]["openable_object"] == ["Openable"]
    assert "LiftObjectTask" not in tasks
    assert any(entry["name"] == "franka_put_and_close_door" for entry in catalogue["environments"])
    monkeypatch.setattr("sys.argv", ["runner", "--mode", "schema"])
    assert cli_runner.main() == 0
    assert json.loads(capsys.readouterr().out) == ArenaEnvGraphSpec.model_json_schema()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(minimal_spec_dict()))
    monkeypatch.setattr("sys.argv", ["runner", "--mode", "validate", "--format", "json", "--env_spec", str(path)])
    assert cli_runner.main() == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    path.write_text("objects: [")
    assert cli_runner.main() == 1
    assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == "input_error"
