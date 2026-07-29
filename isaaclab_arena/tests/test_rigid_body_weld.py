# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for merging a multi-part USD asset into a single rigid body."""

from __future__ import annotations

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True

SIMREADY_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady"
# A bottle and its cap, held together by one fixed joint.
SIMREADY_BOTTLE_USD = f"{SIMREADY_URL}/Hospital/Cleaners/Disinfectant_B01/sm_disinfectant_b01_01.usd"
# A cabinet with doors on hinges.
SIMREADY_CABINET_USD = f"{SIMREADY_URL}/Residential/Kitchen/Cabinets/Cabinet_D01/sm_fixture_cabinet_d01_01.usd"
# SimReady props only have physics once this variant is selected.
SIMREADY_VARIANTS = {"Physics": "physics"}


def _export(stage, directory, name: str) -> str:
    path = str(directory / f"{name}.usd")
    assert stage.Export(path), f"failed to export {name}"
    return path


def _test_weld_usd_rigid_bodies(simulation_app):
    import tempfile
    from pathlib import Path

    from pxr import Usd

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_utils import detect_object_type
    from isaaclab_arena.tests.utils.usd_stages import add_body, hinged_bodies_stage, new_stage, welded_bodies_stage
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure
    from isaaclab_arena.utils.usd.rigid_body_weld import weld_usd_rigid_bodies

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)

        # Bodies held together by a fixed joint become one body, like a bottle and its cap.
        source = _export(welded_bodies_stage(), directory, "welded")
        welded_path = weld_usd_rigid_bodies(source)
        assert welded_path != source, "expected a new file for bodies tied by a fixed joint"
        structure = get_physics_structure(Usd.Stage.Open(welded_path))
        assert structure.rigid_body_paths == ("/Prop",), structure.rigid_body_paths
        assert not structure.joints, "the joints should be switched off"
        assert detect_object_type(usd_path=welded_path) == ObjectType.RIGID

        # The new file points at the source instead of copying its geometry, so it stays small.
        assert Path(welded_path).stat().st_size < 8192

        # Asking twice returns the same cached file.
        assert weld_usd_rigid_bodies(source) == welded_path

        # An asset whose parts can move is left alone.
        hinged = _export(hinged_bodies_stage(), directory, "hinged")
        assert weld_usd_rigid_bodies(hinged) == hinged

        # So is an asset that already has exactly one rigid body.
        single_stage = new_stage()
        add_body(single_stage, "body_01")
        single = _export(single_stage, directory, "single")
        assert weld_usd_rigid_bodies(single) == single

    return True


def _test_weld_simready_asset(simulation_app):
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_utils import detect_object_type
    from isaaclab_arena.utils.usd.rigid_body_weld import weld_usd_rigid_bodies

    # Without the weld, the bottle has two bodies and Isaac Lab refuses to spawn it.
    assert detect_object_type(usd_path=SIMREADY_BOTTLE_USD, variants=SIMREADY_VARIANTS) == ObjectType.RIGID
    bottle_path = weld_usd_rigid_bodies(SIMREADY_BOTTLE_USD, SIMREADY_VARIANTS)
    assert bottle_path != SIMREADY_BOTTLE_USD, "expected the bottle and its cap to be merged"
    assert detect_object_type(usd_path=bottle_path, variants=SIMREADY_VARIANTS) == ObjectType.RIGID

    # The cabinet doors swing, so it stays an articulation and is left alone.
    assert detect_object_type(usd_path=SIMREADY_CABINET_USD, variants=SIMREADY_VARIANTS) == ObjectType.ARTICULATION
    assert weld_usd_rigid_bodies(SIMREADY_CABINET_USD, SIMREADY_VARIANTS) == SIMREADY_CABINET_USD

    return True


def test_weld_usd_rigid_bodies():
    result = run_simulation_app_function(
        _test_weld_usd_rigid_bodies,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_weld_simready_asset():
    result = run_simulation_app_function(
        _test_weld_simready_asset,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_weld_usd_rigid_bodies()
    test_weld_simready_asset()
