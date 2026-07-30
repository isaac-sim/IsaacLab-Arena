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


def _open_simready(usd_path: str):
    """Open a SimReady asset with its physics turned on."""
    from pxr import Usd

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    stage = Usd.Stage.Open(usd_path)
    apply_usd_variant_selections(stage, SIMREADY_VARIANTS)
    return stage


def _bounding_box(stage):
    """The space the asset takes up, as ``(min, max)``."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    return tuple(box.GetMin()), tuple(box.GetMax())


def _collision_meshes(stage) -> dict[str, int]:
    """Map each collision mesh's name to how many points it has."""
    from pxr import Usd, UsdGeom, UsdPhysics

    meshes = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            name = prim.GetName()
            meshes[name] = len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])
    return meshes


def _test_weld_keeps_shape_and_mass(simulation_app):
    """The merged bottle fills the same space, keeps its collision meshes, and weighs the same."""
    from isaaclab_arena.utils.usd.mass_properties import read_mass_properties
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure
    from isaaclab_arena.utils.usd.rigid_body_weld import weld_usd_rigid_bodies

    source_stage = _open_simready(SIMREADY_BOTTLE_USD)
    welded_stage = _open_simready(weld_usd_rigid_bodies(SIMREADY_BOTTLE_USD, SIMREADY_VARIANTS))

    # Nothing was moved or thrown away, so the bottle still fills the same space.
    assert _bounding_box(welded_stage) == _bounding_box(source_stage)

    # Both parts still collide, with the same meshes as before.
    assert _collision_meshes(welded_stage) == _collision_meshes(source_stage)
    assert len(_collision_meshes(welded_stage)) == 2

    parts = [
        read_mass_properties(source_stage.GetPrimAtPath(path))
        for path in get_physics_structure(source_stage).rigid_body_paths
    ]
    assert len(parts) == 2 and all(part is not None for part in parts)

    merged = read_mass_properties(welded_stage.GetPrimAtPath("/Prop"))
    assert merged is not None, "the merged bottle should say what it weighs"

    # The cap and the bottle together weigh what they weighed apart.
    total_mass = sum(part.mass for part in parts)
    assert abs(merged.mass - total_mass) < 1e-9, f"{merged.mass} != {total_mass}"

    # The centre of mass sits between the parts, pulled towards the heavier one.
    for axis in range(3):
        expected = sum(part.mass * part.center_of_mass[axis] for part in parts) / total_mass
        assert abs(merged.center_of_mass[axis] - expected) < 1e-6, f"axis {axis}: {merged.center_of_mass}"

    assert all(moment > 0.0 for moment in merged.diagonal_inertia), merged.diagonal_inertia

    return True


def _test_welded_mass_reaches_physx(simulation_app):
    """PhysX gives the spawned bottle the mass and centre of mass the merged asset asked for."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from isaaclab_arena.utils.usd.mass_properties import read_mass_properties
    from isaaclab_arena.utils.usd.rigid_body_weld import weld_usd_rigid_bodies

    welded_path = weld_usd_rigid_bodies(SIMREADY_BOTTLE_USD, SIMREADY_VARIANTS)
    # Hold on to the stage, or the prim read from it goes stale.
    welded_stage = _open_simready(welded_path)
    expected = read_mass_properties(welded_stage.GetPrimAtPath("/Prop"))

    simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    bottle = RigidObject(
        RigidObjectCfg(
            prim_path="/World/bottle",
            spawn=sim_utils.UsdFileCfg(usd_path=welded_path, variants=SIMREADY_VARIANTS),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )
    simulation.reset()

    # One body, so Isaac Lab accepts the bottle as a rigid object at all.
    assert bottle.body_names == ["bottle"], bottle.body_names

    mass = bottle.data.body_mass.torch.flatten().cpu().tolist()
    center_of_mass = bottle.data.body_com_pos_b.torch.flatten().cpu().tolist()
    assert abs(mass[0] - expected.mass) < 1e-9, f"{mass[0]} != {expected.mass}"
    for axis in range(3):
        assert abs(center_of_mass[axis] - expected.center_of_mass[axis]) < 1e-6, center_of_mass

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


def test_weld_keeps_shape_and_mass():
    result = run_simulation_app_function(
        _test_weld_keeps_shape_and_mass,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_welded_mass_reaches_physx():
    result = run_simulation_app_function(
        _test_welded_mass_reaches_physx,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_weld_usd_rigid_bodies()
    test_weld_simready_asset()
    test_weld_keeps_shape_and_mass()
    test_welded_mass_reaches_physx()
