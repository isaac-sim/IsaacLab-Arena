# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Service geometry exposes stock Arena contracts without asset generation during discovery."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.agentic_environment_generation.catalogues import (
    RelationCatalogue,
    build_asset_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.semantic_validation import (
    collect_semantic_validation_issues,
    validate_authoring_spec,
)
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict
from isaaclab_arena_environments.return_to_service import asset_adapters
from isaaclab_arena_environments.return_to_service.assets import PreparedAssets


class _Registry:
    def __init__(self, entries):
        self.entries = entries

    def get_all_keys(self):
        return list(self.entries)

    def get_asset_by_name(self, name):
        return self.entries[name]

    def get_task_by_name(self, name):
        return self.entries[name]


def test_service_discovery_and_small_graph_do_not_prepare_assets(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Discovery must not prepare or generate service assets")

    monkeypatch.setattr(asset_adapters, "prepare_assets", fail)
    catalogue = build_asset_catalogue()
    assert "return_to_service_bench" in {entry["name"] for entry in catalogue.backgrounds}
    entries = {entry["name"]: entry for entry in catalogue.objects}
    assert "Pressable" in entries["return_to_service_button"]["provides"]
    assert "Openable" in entries["return_to_service_case"]["provides"]
    assert "battery" in entries["return_to_service_component"]["parameters"]["component"]["enum"]
    example = (
        Path(__file__).parents[2]
        / "isaaclab_arena_environments/return_to_service/authoring_examples/battery_in_bin.yaml"
    )
    graph = load_env_graph_spec_dict(example)
    assert graph["embodiment"]["registry_name"] == "droid_differential_ik"
    assert graph["background"]["registry_name"] == "return_to_service_bench"
    report = validate_authoring_spec(graph)
    assert report["valid"], report


def test_joint_adapters_preserve_scene_identity_actuators_and_reset_state(monkeypatch, tmp_path):
    # Articulation configs need a source path, but constructing them does not start a simulator.
    source = tmp_path / "articulation.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    prepared = SimpleNamespace(button_usd=lambda: source, case_usd=lambda: source)
    monkeypatch.setattr(asset_adapters, "prepare_assets", lambda root: prepared)
    button = asset_adapters.ServiceButton(instance_name="test_button")
    assert isinstance(button, Pressable)
    assert button.pressable_joint_name == "press"
    assert button.get_scene_key() == "test_button"
    assert button.object_cfg.actuators["spring"].stiffness == 180.0
    assert button.object_cfg.actuators["spring"].damping == 1.5
    assert button.object_cfg.init_state.joint_pos == {"press": 0.0}
    assert button.object_cfg.init_state.joint_vel == {"press": 0.0}
    case = asset_adapters.ServiceCase(instance_name="case")
    assert isinstance(case, Openable)
    assert case.object_cfg.actuators["passive"].joint_names_expr == ["hinge", "latch"]
    assert case.object_cfg.actuators["passive"].damping == 0.08
    assert case.object_cfg.init_state.joint_pos == {"hinge": -1.8, "latch": 1.4}
    assert case.object_cfg.init_state.joint_vel == {"hinge": 0.0, "latch": 0.0}
    latch = case.for_joint("latch")
    assert latch.openable_joint_name == "latch"
    assert case.openable_joint_name == "hinge"
    assert latch.object_cfg is case.object_cfg
    assert latch.get_scene_key() == case.get_scene_key()


def test_authored_lighting_variations_compose_cached_opinions_without_source_edits(monkeypatch, tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdLux

    source = tmp_path / "lighting.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    dome = UsdLux.DomeLight.Define(stage, "/Asset/StudioDome")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.92, 1.0))
    stage.GetRootLayer().Save()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    prepared = PreparedAssets(tmp_path, cache, {"assets": {"lighting": {"file": "lighting.usda"}}})
    monkeypatch.setattr(asset_adapters, "prepare_assets", lambda root: prepared)
    lighting = asset_adapters.ServiceLighting(instance_name="lighting")
    baseline_path = lighting.usd_path
    assert not any(variation.enabled for variation in lighting.get_variations())
    assert {variation.name for variation in lighting.get_variations()} == {"intensity", "color"}
    lighting.set_intensity(800.0)
    lighting.set_color((0.3, 0.4, 0.5))
    assert lighting.usd_path != baseline_path
    assert lighting.object_cfg.spawn.usd_path == lighting.usd_path
    edited = Usd.Stage.Open(lighting.usd_path)
    edited_dome = UsdLux.DomeLight(edited.GetPrimAtPath("/Asset/StudioDome"))
    assert edited_dome.GetIntensityAttr().Get() == 800.0
    assert tuple(edited_dome.GetColorAttr().Get()) == pytest.approx((0.3, 0.4, 0.5))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize(
    "asset_type,value",
    [
        (asset_adapters.ServiceComponent, "bench"),
        (asset_adapters.ServiceFixture, "battery"),
        (asset_adapters.ServiceInstrument, "battery"),
    ],
)
def test_adapters_reject_incompatible_geometry_before_preparing_assets(monkeypatch, asset_type, value):
    def fail(*args, **kwargs):
        raise AssertionError("Asset preparation must not run for an invalid category")

    monkeypatch.setattr(asset_adapters, "prepare_assets", fail)
    with pytest.raises(AssertionError, match="Unknown"):
        asset_type(component=value)


@pytest.mark.parametrize("subject", ["fixture", "case"])
def test_region_placement_rejects_nonrigid_registered_service_subjects_without_constructing(monkeypatch, subject):
    from isaaclab_arena.tasks.place_in_region_task import PlaceInRegionTask
    from isaaclab_arena_environments.return_to_service.asset_adapters import (
        ServiceCase,
        ServiceComponent,
        ServiceFixture,
    )

    def fail(*args, **kwargs):
        raise AssertionError("Static validation must not construct service assets")

    for asset_class in (ServiceComponent, ServiceFixture, ServiceCase):
        monkeypatch.setattr(asset_class, "__new__", staticmethod(fail))
    classes = {"part": ServiceComponent, "fixture": ServiceFixture, "case": ServiceCase}
    catalogue = build_asset_catalogue(_Registry(classes))
    task_catalogue = build_task_catalogue(_Registry({"PlaceInRegionTask": PlaceInRegionTask}))
    data = {
        "objects": [{"id": name, "registry_name": name} for name in classes],
        "task": {
            "subtasks": [{
                "kind": "PlaceInRegionTask",
                "params": {
                    "subject": subject,
                    "destination": "fixture",
                    "region_bounds": [[-1, -1, -1], [1, 1, 1]],
                },
            }]
        },
    }
    issues = collect_semantic_validation_issues(data, catalogue, task_catalogue, RelationCatalogue())
    issue = next(issue for issue in issues if issue.path == "/task/subtasks/0/params/subject")
    assert issue.code == "missing_capability"
    assert issue.expected == {"capabilities": ["rigid"]}
    assert issue.compatible_choices == ["part"]
    data["task"]["subtasks"][0]["params"]["subject"] = "part"
    issues = collect_semantic_validation_issues(data, catalogue, task_catalogue, RelationCatalogue())
    assert not any(issue.path.startswith("/task/") for issue in issues)
    # These interfaces certify the declared physics type, not collision-shape compatibility.
    part = next(entry for entry in catalogue.objects if entry["name"] == "part")
    assert "rigid" in part["provides"]


def test_droid_battery_in_bin_example_passes_declared_contract_validation(monkeypatch):
    import yaml
    from pathlib import Path

    from isaaclab_arena_environments.return_to_service import asset_adapters

    def fail(*args, **kwargs):
        raise AssertionError("Example validation must not prepare or instantiate assets")

    monkeypatch.setattr(asset_adapters, "prepare_assets", fail)
    monkeypatch.setattr(Asset, "__init__", fail)
    path = (
        Path(__file__).resolve().parents[2]
        / "isaaclab_arena_environments/return_to_service/authoring_examples/battery_in_bin.yaml"
    )
    data = yaml.safe_load(path.read_text())
    assert data["embodiment"]["registry_name"] == "droid_differential_ik"
    report = validate_authoring_spec(data)
    assert report["valid"], report["issues"]
