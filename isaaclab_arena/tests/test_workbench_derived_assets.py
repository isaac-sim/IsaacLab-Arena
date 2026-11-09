# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registration checks for the derived workbench asset set."""

import importlib.resources
import json

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.sources.derived import DerivedUsdSource


def _test_workbench_asset_classes_keep_roles_out_of_sources(simulation_app):
    from isaaclab_arena_environments.workbench_cleanup import assets

    classes = [
        assets.WorkbenchCleanupWorkbench,
        assets.WorkbenchCleanupWhiteBin,
        assets.WorkbenchCleanupSocketSet,
        assets.WorkbenchCleanupOscilloscope,
        assets.WorkbenchCleanupSlottedScrewdriver,
        assets.WorkbenchCleanupSmallWrench,
    ]
    registry = AssetRegistry()
    assert all(registry.get_asset_by_name(asset.registry_name) is asset for asset in classes)
    assert all(isinstance(asset.usd_path, DerivedUsdSource) for asset in classes)
    assert assets.WorkbenchCleanupSocketSet.object_type is ObjectType.ARTICULATION
    assert assets.WorkbenchCleanupSocketSet.openable_joint_name == "case_lid"
    assert assets.WorkbenchCleanupOscilloscope.object_type is ObjectType.ARTICULATION
    assert assets.WorkbenchCleanupOscilloscope.pressable_joint_name == "joint_button_01_joint_00"
    assert assets.WorkbenchCleanupWhiteBin.object_type is ObjectType.RIGID
    assert "container" in assets.WorkbenchCleanupWhiteBin.tags
    assert "background" in assets.WorkbenchCleanupWorkbench.tags

    return True


def test_workbench_asset_classes_keep_roles_out_of_sources():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_asset_classes_keep_roles_out_of_sources)


def _test_workbench_articulations_accept_scene_joint_metadata(simulation_app, monkeypatch):
    from isaaclab_arena_environments.workbench_cleanup import assets

    monkeypatch.setattr(assets.WorkbenchCleanupSocketSet.usd_path, "resolve", lambda: "/cache/socket_set.usda")
    monkeypatch.setattr(assets.WorkbenchCleanupOscilloscope.usd_path, "resolve", lambda: "/cache/oscilloscope.usda")

    socket_set = assets.WorkbenchCleanupSocketSet(initial_joint_positions={"case_lid": 1.5})
    oscilloscope = assets.WorkbenchCleanupOscilloscope()

    assert socket_set.object_cfg.init_state.joint_pos == {"case_lid": 1.5}
    assert socket_set.openable_joint_name == "case_lid"
    assert oscilloscope.pressable_joint_name == "joint_button_01_joint_00"

    return True


def test_workbench_articulations_accept_scene_joint_metadata(monkeypatch):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_workbench_articulations_accept_scene_joint_metadata, monkeypatch=monkeypatch
    )


def _test_workbench_derivation_recipes_match_source_identities(simulation_app):
    from isaaclab_arena_environments.workbench_cleanup import assets

    classes = [
        assets.WorkbenchCleanupWorkbench,
        assets.WorkbenchCleanupWhiteBin,
        assets.WorkbenchCleanupSocketSet,
        assets.WorkbenchCleanupOscilloscope,
        assets.WorkbenchCleanupSlottedScrewdriver,
        assets.WorkbenchCleanupSmallWrench,
    ]
    for asset in classes:
        source = asset.usd_path
        package_path = source.derivation_spec.removeprefix("package://")
        package, relative_path = package_path.split("/", 1)
        recipe = json.loads(importlib.resources.files(package).joinpath(relative_path).read_bytes())
        assert recipe["source_identity"] == source.source.identity()
        assert recipe["operation"] == "isaaclab_arena.assets.derivation.pipeline:derive_asset"
        assert recipe["steps"]

    return True


def test_workbench_derivation_recipes_match_source_identities():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_derivation_recipes_match_source_identities)


def test_normal_environment_discovery_registers_workbench_assets_in_a_fresh_process():
    import subprocess
    import sys

    code = """
import sys
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.sources.derived import DerivedUsdSource
from isaaclab_arena.assets.sources.simready import SimReadyCentralPackageSource

registry = AssetRegistry()
assert not registry.is_registered('workbench_cleanup_workbench', ensure_loaded=False)
assert 'isaaclab_arena_environments.workbench_cleanup.assets' not in sys.modules

def unexpected_resolution(self):
    raise AssertionError('Registration must not download or derive assets')

DerivedUsdSource.resolve = unexpected_resolution
SimReadyCentralPackageSource.resolve = unexpected_resolution

from isaaclab_arena_environments.cli import ensure_environments_registered
ensure_environments_registered()

expected = {
    'workbench_cleanup_workbench', 'workbench_cleanup_white_bin',
    'workbench_cleanup_socket_set', 'workbench_cleanup_oscilloscope',
    'workbench_cleanup_slotted_screwdriver', 'workbench_cleanup_small_wrench',
}
for name in expected:
    asset_type = registry.get_asset_by_name(name)
    assert asset_type.__module__ == 'isaaclab_arena_environments.workbench_cleanup.assets'

assert 'isaaclab_arena_environments.workbench_cleanup.prebuild_assets' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
