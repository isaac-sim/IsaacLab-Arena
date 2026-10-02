# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject malformed asset contracts before simulation or GPU measurements begin."""

import json

import pytest

from isaaclab_arena_environments.return_to_service.assets import prepare_assets, validate_asset_manifest


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "part.usda").write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    manifest = {
        "schema_version": 1,
        "units": "meters",
        "up_axis": "Z",
        "assets": {
            "part": {
                "file": "part.usda",
                "root_prim": "/Asset",
                "mass_kg": 0.1,
                "collision_count": 1,
                "bounds_min": [-0.1, -0.1, 0],
                "bounds_max": [0.1, 0.1, 0.1],
                "affordances": {
                    "battery_socket": [0, 0, 0.05],
                    "grasp": {"position_xyz": [0, 0, 0.08], "rotation_xyzw": [0, 0, 0, 1]},
                    "interior_bounds": [[-0.05, -0.05, 0.01], [0.05, 0.05, 0.09]],
                    "status_paths": {"idle": "/Asset/display_idle"},
                },
            }
        },
    }
    return tmp_path, manifest


def test_valid_bundle_is_loaded_without_authoring_physics(bundle):
    root, manifest = bundle
    (root / "manifest.json").write_text(json.dumps(manifest))
    prepared = prepare_assets(root)
    assert prepared.source_path("part") == root / "part.usda"
    assert list(prepared.cache_root.iterdir()) == []


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("units",), "centimeters", "meters"),
        (("assets", "part", "mass_kg"), -1, "mass"),
        (("assets", "part", "mass_kg"), True, "mass"),
        (("assets", "part", "collision_count"), -1, "collider"),
        (("assets", "part", "bounds_min"), [0.2, -0.1, 0], "Reversed"),
        (("assets", "part", "bounds_max"), [0.1, 0.1, 0], "Empty"),
        (("assets", "part", "bounds_max"), [0.1, 0.1, float("inf")], "Non-finite"),
        (("assets", "part", "affordances", "battery_socket"), [0, 0, float("nan")], "Non-finite"),
        (("assets", "part", "affordances", "battery_socket"), [0, 0], "coordinates"),
        (("assets", "part", "affordances", "grasp", "rotation_xyzw"), [0, 0, 0, 0], "normalized"),
        (("assets", "part", "affordances", "interior_bounds"), [[0, 0, 0], [-1, 1, 1]], "Reversed"),
        (("assets", "part", "affordances", "status_paths"), {"idle": "/OtherRoot/display"}, "beneath"),
        (("assets", "part", "affordances", "cavity_cylinder"), {"radius": -0.01}, "positive"),
        (("assets", "part", "affordances", "cavity_cylinder"), {"x_range": [0.1, -0.1]}, "ordered"),
        (("assets", "part", "root_prim"), "Asset", "root prim"),
        (("assets", "part", "file"), "../part.usda", "escapes"),
        (("assets", "part", "file"), "/tmp/part.usda", "escapes"),
    ],
)
def test_invalid_manifest_fails_before_measurement(bundle, path, value, message):
    root, manifest = bundle
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError, match=message):
        validate_asset_manifest(manifest, root)


def test_missing_geometry_fields_cannot_silently_become_visual_only(bundle):
    root, manifest = bundle
    del manifest["assets"]["part"]["collision_count"]
    with pytest.raises(AssertionError, match="Incomplete geometry"):
        validate_asset_manifest(manifest, root)


def test_symlink_source_cannot_escape_asset_bundle(bundle, tmp_path_factory):
    root, manifest = bundle
    outside = tmp_path_factory.mktemp("external") / "outside.usda"
    outside.write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    (root / "linked.usda").symlink_to(outside)
    manifest["assets"]["part"]["file"] = "linked.usda"
    with pytest.raises(AssertionError, match="escapes"):
        validate_asset_manifest(manifest, root)


def test_visual_lighting_and_planar_worktop_affordance_are_valid(bundle):
    root, manifest = bundle
    manifest["assets"]["lighting"] = {"file": "part.usda", "root_prim": "/Asset", "affordances": {}}
    manifest["assets"]["part"]["affordances"]["usable_bounds"] = [[-0.1, -0.1, 0], [0.1, 0.1, 0]]
    validate_asset_manifest(manifest, root)


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"static_friction_effort_nm": 0.012}, "Incomplete"),
        ({"static_friction_effort_nm": True, "dynamic_friction_effort_nm": 0.01}, "numbers"),
        ({"static_friction_effort_nm": 0.012, "dynamic_friction_effort_nm": -0.01}, "0 <= dynamic"),
        ({"static_friction_effort_nm": 0.012, "dynamic_friction_effort_nm": 0.02}, "0 <= dynamic"),
        ({"static_friction_effort_nm": float("inf"), "dynamic_friction_effort_nm": 0.01}, "Non-finite"),
        ({"angle_limits_degrees": [100, 0]}, "ordered"),
        ({"angle_limits_degrees": [0, 0]}, "ordered"),
        ({"angle_limits_degrees": [0]}, "coordinates"),
    ],
)
def test_invalid_joint_mechanics_are_rejected(bundle, settings, message):
    root, manifest = bundle
    manifest["assets"]["part"]["affordances"]["latch"] = settings
    with pytest.raises(AssertionError, match=message):
        validate_asset_manifest(manifest, root)


def test_passive_joint_mechanics_have_explicit_units(bundle):
    root, manifest = bundle
    manifest["assets"]["part"]["affordances"]["latch"] = {
        "angle_limits_degrees": [0, 100],
        "static_friction_effort_nm": 0.012,
        "dynamic_friction_effort_nm": 0.01,
    }
    validate_asset_manifest(manifest, root)
