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


def _test_merge_to_one_rigid_body_if_needed(simulation_app):
    import tempfile
    from pathlib import Path

    from pxr import Usd

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_utils import detect_object_type
    from isaaclab_arena.tests.utils.usd_stages import add_body, fixed_joint_bodies_stage, hinged_bodies_stage, new_stage
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure_from_usd
    from isaaclab_arena.utils.usd.rigid_body_merge import merge_to_one_rigid_body_if_needed

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)

        # Bodies held together by a fixed joint become one body, like a bottle and its cap.
        source = _export(fixed_joint_bodies_stage(), directory, "fixed_joint")
        merged_path = merge_to_one_rigid_body_if_needed(source)
        assert merged_path != source, "expected a new file for bodies tied by a fixed joint"
        structure = get_physics_structure_from_usd(Usd.Stage.Open(merged_path))
        assert structure.rigid_body_paths == ("/Prop",), structure.rigid_body_paths
        assert not structure.joints, "the joints should be switched off"
        assert detect_object_type(usd_path=merged_path) == ObjectType.RIGID

        # The new file points at the source instead of copying its geometry, so it stays small.
        assert Path(merged_path).stat().st_size < 8192

        # An asset whose parts can move is left alone.
        hinged = _export(hinged_bodies_stage(), directory, "hinged")
        assert merge_to_one_rigid_body_if_needed(hinged) == hinged

        # So is an asset that already has exactly one rigid body.
        single_stage = new_stage()
        add_body(single_stage, "body_01")
        single = _export(single_stage, directory, "single")
        assert merge_to_one_rigid_body_if_needed(single) == single

    return True


def _test_merge_remembers_its_answers(simulation_app):
    """An asset is only merged once: the answer is kept in memory, the merged file on disk."""
    import tempfile
    from pathlib import Path

    from isaaclab_arena.tests.utils.usd_stages import fixed_joint_bodies_stage
    from isaaclab_arena.utils.usd import rigid_body_merge
    from isaaclab_arena.utils.usd.rigid_body_merge import merge_to_one_rigid_body_if_needed

    with tempfile.TemporaryDirectory() as directory_name:
        source = _export(fixed_joint_bodies_stage(), Path(directory_name), "remembered")
        merged_path = merge_to_one_rigid_body_if_needed(source)
        written_at = Path(merged_path).stat().st_mtime_ns

        # Asking again gives the same file back.
        assert merge_to_one_rigid_body_if_needed(source) == merged_path

        # With the answer forgotten, the file already on disk is used instead of being written again.
        rigid_body_merge._merged_paths.clear()
        assert merge_to_one_rigid_body_if_needed(source) == merged_path
        assert Path(merged_path).stat().st_mtime_ns == written_at, "the merged asset was written twice"

    return True


def _test_auto_detected_object_spawns_from_merged_asset(simulation_app):
    """An object that works out its own type gets a path it can actually spawn from."""
    import tempfile
    from pathlib import Path

    from pxr import Usd

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.tests.utils.usd_stages import fixed_joint_bodies_stage, hinged_bodies_stage
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure_from_usd

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)

        # Calling an asset rigid is only true once its parts are merged, so the object has to
        # come away pointing at the merged file rather than the one it was given.
        source = _export(fixed_joint_bodies_stage(), directory, "auto_fixed_joint")
        merged = Object(name="bottle", prim_path="{ENV_REGEX_NS}/bottle", object_type=None, usd_path=source)
        assert merged.object_type == ObjectType.RIGID
        assert merged.usd_path != source, "the object still points at the asset with two bodies"
        assert get_physics_structure_from_usd(Usd.Stage.Open(merged.usd_path)).rigid_body_paths == ("/Prop",)

        # An articulation is left with the asset it was given, since nothing was merged.
        hinged = _export(hinged_bodies_stage(), directory, "auto_hinged")
        swinging = Object(name="cabinet", prim_path="{ENV_REGEX_NS}/cabinet", object_type=None, usd_path=hinged)
        assert swinging.object_type == ObjectType.ARTICULATION
        assert swinging.usd_path == hinged

    return True


def _open_simready(usd_path: str):
    """Open a SimReady asset with its physics turned on."""
    from pxr import Usd

    from isaaclab_arena.utils.usd_helpers import apply_usd_variant_selections

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


def _test_merge_keeps_shape_and_mass(simulation_app):
    """The merged bottle fills the same space, keeps its collision meshes, and weighs the same."""
    from isaaclab_arena.utils.usd.mass_properties import read_mass_properties
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure_from_usd
    from isaaclab_arena.utils.usd.rigid_body_merge import merge_to_one_rigid_body_if_needed

    source_stage = _open_simready(SIMREADY_BOTTLE_USD)
    merged_stage = _open_simready(merge_to_one_rigid_body_if_needed(SIMREADY_BOTTLE_USD, SIMREADY_VARIANTS))

    # Nothing was moved or thrown away, so the bottle still fills the same space.
    assert _bounding_box(merged_stage) == _bounding_box(source_stage)

    # Both parts still collide, with the same meshes as before.
    assert _collision_meshes(merged_stage) == _collision_meshes(source_stage)
    assert len(_collision_meshes(merged_stage)) == 2

    parts = [
        read_mass_properties(source_stage.GetPrimAtPath(path))
        for path in get_physics_structure_from_usd(source_stage).rigid_body_paths
    ]
    assert len(parts) == 2 and all(part is not None for part in parts)

    merged = read_mass_properties(merged_stage.GetPrimAtPath("/Prop"))
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


def _test_merged_mass_reaches_physx(simulation_app):
    """PhysX gives the spawned bottle the mass and centre of mass the merged asset asked for."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from isaaclab_arena.utils.usd.mass_properties import read_mass_properties
    from isaaclab_arena.utils.usd.rigid_body_merge import merge_to_one_rigid_body_if_needed

    merged_path = merge_to_one_rigid_body_if_needed(SIMREADY_BOTTLE_USD, SIMREADY_VARIANTS)
    # Hold on to the stage, or the prim read from it goes stale.
    merged_stage = _open_simready(merged_path)
    expected = read_mass_properties(merged_stage.GetPrimAtPath("/Prop"))

    simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    bottle = RigidObject(
        RigidObjectCfg(
            prim_path="/World/bottle",
            spawn=sim_utils.UsdFileCfg(usd_path=merged_path, variants=SIMREADY_VARIANTS),
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


def _test_merge_simready_asset(simulation_app):
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_utils import detect_object_type
    from isaaclab_arena.utils.usd.rigid_body_merge import merge_to_one_rigid_body_if_needed

    # Without merging, the bottle has two bodies and Isaac Lab refuses to spawn it.
    assert detect_object_type(usd_path=SIMREADY_BOTTLE_USD, variants=SIMREADY_VARIANTS) == ObjectType.RIGID
    bottle_path = merge_to_one_rigid_body_if_needed(SIMREADY_BOTTLE_USD, SIMREADY_VARIANTS)
    assert bottle_path != SIMREADY_BOTTLE_USD, "expected the bottle and its cap to be merged"
    assert detect_object_type(usd_path=bottle_path, variants=SIMREADY_VARIANTS) == ObjectType.RIGID

    # The cabinet doors swing, so it stays an articulation and is left alone.
    assert detect_object_type(usd_path=SIMREADY_CABINET_USD, variants=SIMREADY_VARIANTS) == ObjectType.ARTICULATION
    assert merge_to_one_rigid_body_if_needed(SIMREADY_CABINET_USD, SIMREADY_VARIANTS) == SIMREADY_CABINET_USD

    return True


def test_merge_to_one_rigid_body_if_needed():
    result = run_simulation_app_function(
        _test_merge_to_one_rigid_body_if_needed,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_merge_remembers_its_answers():
    result = run_simulation_app_function(
        _test_merge_remembers_its_answers,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_merge_simready_asset():
    result = run_simulation_app_function(
        _test_merge_simready_asset,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_auto_detected_object_spawns_from_merged_asset():
    result = run_simulation_app_function(
        _test_auto_detected_object_spawns_from_merged_asset,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_merge_keeps_shape_and_mass():
    result = run_simulation_app_function(
        _test_merge_keeps_shape_and_mass,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_merged_mass_reaches_physx():
    result = run_simulation_app_function(
        _test_merged_mass_reaches_physx,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_merge_to_one_rigid_body_if_needed()
    test_merge_remembers_its_answers()
    test_auto_detected_object_spawns_from_merged_asset()
    test_merge_simready_asset()
    test_merge_keeps_shape_and_mass()
    test_merged_mass_reaches_physx()
