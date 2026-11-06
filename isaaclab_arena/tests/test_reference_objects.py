# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pathlib
import torch
import tqdm
import traceback
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

NUM_STEPS = 100
HEADLESS = True
OPEN_STEP = NUM_STEPS // 2


def background_from_usd_path(name: str, usd_path: pathlib.Path, initial_pose: Pose, object_min_z: float = -0.2):

    from isaaclab_arena.assets.background import Background

    class ObjectReferenceTestKitchenBackground(Background):
        """
        Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
        """

        def __init__(self):
            super().__init__(
                name=name,
                tags=["background"],
                usd_path=str(usd_path),
                initial_pose=initial_pose,
                object_min_z=object_min_z,
            )

    return ObjectReferenceTestKitchenBackground()


def _object_reference_with_cached_bbox(parent_pose: Pose | None, relative_pose: Pose, bbox: AxisAlignedBoundingBox):
    """Construct an ObjectReference around cached geometry without opening a USD."""
    from isaaclab_arena.assets.object_reference import ObjectReference

    obj_ref = ObjectReference.__new__(ObjectReference)
    obj_ref.parent_asset = SimpleNamespace(initial_pose=parent_pose)
    obj_ref.initial_pose_relative_to_parent = relative_pose
    obj_ref._bounding_box = bbox
    return obj_ref


def test_object_reference_world_bbox_applies_parent_yaw():
    """Parent yaw, not the prim's relative yaw, rotates the already-local referenced bbox."""
    yaw_90 = (0.0, 0.0, 2**-0.5, 2**-0.5)
    obj_ref = _object_reference_with_cached_bbox(
        parent_pose=Pose(position_xyz=(10.0, 0.0, 0.0), rotation_xyzw=yaw_90),
        relative_pose=Pose(position_xyz=(1.0, 2.0, 0.0), rotation_xyzw=yaw_90),
        bbox=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.1, 0.05)),
    )

    world_bbox = obj_ref.get_world_bounding_box()

    assert torch.allclose(world_bbox.min_point, torch.tensor([[7.9, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(world_bbox.max_point, torch.tensor([[8.0, 1.2, 0.05]]), atol=1e-6)


@pytest.mark.parametrize(
    "parent_pose,expected_lower,expected_upper",
    [
        (None, (1.0, 2.0, 0.0), (2.6, 2.8, 0.1)),
        (Pose((10.0, 0.0, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5)), (7.2, 1.0, 0.0), (8.0, 2.6, 0.1)),
    ],
)
def test_reference_anchor_bounds_do_not_reapply_prim_rotation(parent_pose, expected_lower, expected_upper):
    from isaaclab_arena.relations.geometry.bounding_box_helpers import build_per_env_bounding_boxes
    from isaaclab_arena.relations.relations import IsAnchor

    reference = _object_reference_with_cached_bbox(
        parent_pose,
        Pose((1.0, 2.0, 0.0), (0, 0, 0.3826834324, 0.9238795325)),
        AxisAlignedBoundingBox((0, 0, 0), (1.6, 0.8, 0.1)),
    )
    reference.name = "counter"
    reference.relations = [IsAnchor()]
    bounds = build_per_env_bounding_boxes([reference], num_envs=2).object_bboxes[reference]
    world_bounds = bounds.translated(reference.get_initial_pose().position_xyz)
    torch.testing.assert_close(world_bounds.min_point, torch.tensor([expected_lower] * 2))
    torch.testing.assert_close(world_bounds.max_point, torch.tensor([expected_upper] * 2))
    torch.testing.assert_close(reference.get_world_bounding_box().min_point, torch.tensor([expected_lower]))
    torch.testing.assert_close(reference.get_world_bounding_box().max_point, torch.tensor([expected_upper]))
    assert reference.get_parent_pose() == (parent_pose if parent_pose is not None else Pose.identity())


def test_object_reference_caches_parent_usd_prim_path(monkeypatch):
    """Resolving the initial pose also caches the parent-USD path for later use."""
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType

    calls = {"open_count": 0}
    obj_ref = ObjectReference.__new__(ObjectReference)
    obj_ref.prim_path = "{ENV_REGEX_NS}/kitchen/counter"
    obj_ref.object_type = ObjectType.BASE
    obj_ref._parent_scale = (1.0, 1.0, 1.0)

    def open_spawned_usd_stage():
        calls["open_count"] += 1
        return SimpleNamespace(GetPrimAtPath=lambda path: object())

    parent = SimpleNamespace(
        usd_path="/tmp/kitchen.usd",
        name="kitchen",
        object_type=ObjectType.BASE,
        open_spawned_usd_stage=open_spawned_usd_stage,
    )

    monkeypatch.setattr(
        ObjectReference,
        "isaaclab_prim_path_to_original_prim_path",
        staticmethod(lambda prim_path, parent_asset, stage: "/Asset/counter"),
    )
    monkeypatch.setattr(
        "isaaclab_arena.assets.object_reference._get_prim_path_in_referenced_usd",
        lambda stage, prim_path: "/World/counter",
    )
    monkeypatch.setattr(
        "isaaclab_arena.assets.object_reference.get_prim_pose_in_default_prim_frame",
        lambda prim, stage, prim_world_delta: Pose(),
    )

    (
        obj_ref._prim_path_in_parent_usd,
        initial_link_world_delta,
        pose,
    ) = obj_ref._get_referenced_prim_path_and_pose_relative_to_parent(parent)

    assert obj_ref.prim_path_in_parent_usd == "/World/counter"
    assert initial_link_world_delta is None
    assert pose == Pose()
    assert calls["open_count"] == 1


def _write_scaled_drawer_articulation(
    path: pathlib.Path,
    joint_type: str,
    collider_scale: float = 1.0,
    drawer_physics_in_variant: bool = False,
    repeat_joint_name: bool = False,
) -> None:
    """Write a cabinet scaled by 2 whose drawer body is driven by one prismatic or revolute joint.

    The drawer origin sits at x = 0.2 in the default prim frame. Its collider spans x in [-0.4, 0.8]
    from that origin, so a rotation's direction changes its bounds.

    Args:
        path: File to write the USD stage to.
        joint_type: "prismatic" or "revolute".
        collider_scale: Scale of the collider the drawer joint targets, relative to the drawer body.
        drawer_physics_in_variant: Author the drawer's rigid body only inside a "Physics" variant that is
            not selected by default, as SimReady assets do.
        repeat_joint_name: Add a second drawer whose joint repeats the first joint's name.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cabinet = UsdGeom.Xform.Define(stage, "/Asset/Cabinet")
    cabinet.AddScaleOp().Set(Gf.Vec3f(2.0))
    UsdPhysics.ArticulationRootAPI.Apply(cabinet.GetPrim())

    base = UsdGeom.Xform.Define(stage, "/Asset/Cabinet/Base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    drawer = UsdGeom.Xform.Define(stage, "/Asset/Cabinet/Drawer")
    drawer.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.0, 0.0))
    if drawer_physics_in_variant:
        physics_variants = root.GetVariantSets().AddVariantSet("Physics")
        physics_variants.AddVariant("none")
        physics_variants.AddVariant("physics")
        physics_variants.SetVariantSelection("physics")
        with physics_variants.GetVariantEditContext():
            UsdPhysics.RigidBodyAPI.Apply(drawer.GetPrim())
        physics_variants.SetVariantSelection("none")
    else:
        UsdPhysics.RigidBodyAPI.Apply(drawer.GetPrim())
    collider = UsdGeom.Cube.Define(stage, "/Asset/Cabinet/Drawer/Collider")
    collider.CreateExtentAttr([Gf.Vec3f(-0.2, -0.1, -0.05), Gf.Vec3f(0.4, 0.1, 0.05)])
    if collider_scale != 1.0:
        collider.AddScaleOp().Set(Gf.Vec3f(collider_scale))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    joint_schema = UsdPhysics.PrismaticJoint if joint_type == "prismatic" else UsdPhysics.RevoluteJoint
    joint = joint_schema.Define(stage, "/Asset/Cabinet/drawer_joint")
    joint.CreateBody0Rel().SetTargets(["/Asset/Cabinet/Base"])
    # Target the collider, as SimReady assets do, rather than the drawer body itself.
    joint.CreateBody1Rel().SetTargets(["/Asset/Cabinet/Drawer/Collider"])
    joint.CreateAxisAttr("X" if joint_type == "prismatic" else "Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    if repeat_joint_name:
        second_drawer = UsdGeom.Xform.Define(stage, "/Asset/Cabinet/SecondDrawer").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(second_drawer)
        second_joint = UsdPhysics.PrismaticJoint.Define(stage, "/Asset/Cabinet/SecondDrawer/drawer_joint")
        second_joint.CreateBody0Rel().SetTargets(["/Asset/Cabinet/Base"])
        second_joint.CreateBody1Rel().SetTargets(["/Asset/Cabinet/SecondDrawer"])
        second_joint.CreateAxisAttr("X")
    stage.GetRootLayer().Save()


def _drawer_reference(
    tmp_path: pathlib.Path,
    joint_type: str,
    joint_position: float,
    spawn_scale: float = 1.0,
    body_name: str = "Drawer",
    **stage_options,
):
    """Return a BASE reference to a body of a cabinet whose drawer is opened to joint_position.

    stage_options are passed to _write_scaled_drawer_articulation.
    """
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType

    usd_path = tmp_path / f"{joint_type}.usda"
    _write_scaled_drawer_articulation(usd_path, joint_type, **stage_options)
    cabinet = Object(
        name="cabinet",
        usd_path=str(usd_path),
        object_type=ObjectType.ARTICULATION,
        scale=(spawn_scale, spawn_scale, spawn_scale),
    )
    cabinet.object_cfg.init_state.joint_pos = {"drawer_.*": joint_position}
    return ObjectReference(
        name="drawer",
        prim_path=f"{{ENV_REGEX_NS}}/cabinet/Cabinet/{body_name}",
        parent_asset=cabinet,
        object_type=ObjectType.BASE,
    )


def _test_articulation_link_reference_follows_initial_prismatic_joint_position(
    simulation_app, tmp_path: pathlib.Path, spawn_scale: float, expected_x: float, expected_min_W, expected_max_W
) -> bool:
    from isaaclab_arena.assets.object_reference import ArticulationBodyReferenceCfg

    drawer = _drawer_reference(tmp_path, "prismatic", 0.3, spawn_scale)

    assert isinstance(drawer.object_cfg, ArticulationBodyReferenceCfg)
    assert drawer.object_cfg.articulation_name == "cabinet"
    assert drawer.object_cfg.body_name == "Drawer"
    assert drawer.get_initial_pose().position_xyz == pytest.approx((expected_x, 0.0, 0.0))
    bounds_W = drawer.get_world_bounding_box()
    torch.testing.assert_close(bounds_W.min_point, torch.tensor([expected_min_W]))
    torch.testing.assert_close(bounds_W.max_point, torch.tensor([expected_max_W]))
    return True


@pytest.mark.parametrize(
    "spawn_scale,expected_x,expected_min_W,expected_max_W",
    [
        # The 0.2 authored drawer offset plus the 0.3 m joint position.
        (1.0, 0.5, (0.1, -0.2, -0.1), (1.3, 0.2, 0.1)),
        # The spawn scale halves the authored offset and collider, but not the joint position.
        (0.5, 0.4, (0.2, -0.1, -0.05), (0.8, 0.1, 0.05)),
    ],
)
def test_articulation_link_reference_follows_initial_prismatic_joint_position(
    tmp_path: pathlib.Path, spawn_scale, expected_x, expected_min_W, expected_max_W
):
    """A drawer reference is placed where the cabinet's initial joint position puts the drawer body."""
    assert run_function_with_persistent_simulation_app(
        _test_articulation_link_reference_follows_initial_prismatic_joint_position,
        headless=HEADLESS,
        tmp_path=tmp_path,
        spawn_scale=spawn_scale,
        expected_x=expected_x,
        expected_min_W=expected_min_W,
        expected_max_W=expected_max_W,
    )


def _test_articulation_link_reference_follows_initial_revolute_joint_angle(
    simulation_app, tmp_path: pathlib.Path
) -> bool:
    drawer = _drawer_reference(tmp_path, "revolute", np.pi / 2)

    initial_pose = drawer.get_initial_pose()
    assert initial_pose.position_xyz == pytest.approx((0.2, 0.0, 0.0))
    assert initial_pose.rotation_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))
    # +90 degrees maps the collider's x span [-0.4, 0.8] onto y.
    bounds_W = drawer.get_world_bounding_box()
    torch.testing.assert_close(bounds_W.min_point, torch.tensor([[0.0, -0.4, -0.1]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(bounds_W.max_point, torch.tensor([[0.4, 0.8, 0.1]]), atol=1e-6, rtol=0)
    return True


def test_articulation_link_reference_follows_initial_revolute_joint_angle(tmp_path: pathlib.Path):
    """A drawer reference turned a quarter turn about +Z rotates its pose and bounds the same way."""
    assert run_function_with_persistent_simulation_app(
        _test_articulation_link_reference_follows_initial_revolute_joint_angle,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def _test_articulation_link_reference_keeps_its_authored_pose_with_every_joint_at_zero(
    simulation_app, tmp_path: pathlib.Path
) -> bool:
    from isaaclab_arena.assets.object_reference import ArticulationBodyReferenceCfg

    # The drawer joint targets a collider scaled relative to the drawer, which posing cannot move rigidly.
    (tmp_path / "closed").mkdir()
    (tmp_path / "open").mkdir()
    closed_drawer = _drawer_reference(tmp_path / "closed", "prismatic", 0.0, collider_scale=0.5)

    assert isinstance(closed_drawer.object_cfg, ArticulationBodyReferenceCfg)
    assert closed_drawer.get_initial_pose().position_xyz == pytest.approx((0.2, 0.0, 0.0))
    with pytest.raises(AssertionError, match="scaled relative to its body"):
        _drawer_reference(tmp_path / "open", "prismatic", 0.3, collider_scale=0.5)
    return True


def test_articulation_link_reference_keeps_its_authored_pose_with_every_joint_at_zero(tmp_path: pathlib.Path):
    """With every joint at zero a link reference keeps its authored pose, even where posing a joint would fail."""
    assert run_function_with_persistent_simulation_app(
        _test_articulation_link_reference_keeps_its_authored_pose_with_every_joint_at_zero,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def _test_articulation_link_reference_to_an_undriven_link_allows_repeated_joint_names(
    simulation_app, tmp_path: pathlib.Path
) -> bool:
    from isaaclab_arena.assets.object_reference import ArticulationBodyReferenceCfg

    base = _drawer_reference(tmp_path, "prismatic", 0.3, body_name="Base", repeat_joint_name=True)

    assert isinstance(base.object_cfg, ArticulationBodyReferenceCfg)
    assert base.object_cfg.body_name == "Base"
    assert base.get_initial_pose().position_xyz == pytest.approx((0.0, 0.0, 0.0))
    return True


def test_articulation_link_reference_to_an_undriven_link_allows_repeated_joint_names(tmp_path: pathlib.Path):
    """A reference to a link that no joint moves builds even when the parent repeats a joint name."""
    assert run_function_with_persistent_simulation_app(
        _test_articulation_link_reference_to_an_undriven_link_allows_repeated_joint_names,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def _test_articulation_link_reference_selects_the_parent_spawn_variants(simulation_app, tmp_path: pathlib.Path) -> bool:
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ArticulationBodyReferenceCfg, ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType

    # Without the Physics variant the drawer has no rigid body, so it would not be posed as a link.
    usd_path = tmp_path / "prismatic.usda"
    _write_scaled_drawer_articulation(usd_path, "prismatic", drawer_physics_in_variant=True)
    # Keep the source layer open, as a stage that already spawned the cabinet would.
    source_stage = Usd.Stage.Open(str(usd_path))
    assert not source_stage.GetPrimAtPath("/Asset/Cabinet/Drawer").HasAPI(UsdPhysics.RigidBodyAPI)

    cabinet = Object(
        name="cabinet",
        usd_path=str(usd_path),
        object_type=ObjectType.ARTICULATION,
        spawn_cfg_addon={"variants": {"Physics": "physics"}},
    )
    cabinet.object_cfg.init_state.joint_pos = {"drawer_.*": 0.3}
    drawer = ObjectReference(
        name="drawer",
        prim_path="{ENV_REGEX_NS}/cabinet/Cabinet/Drawer",
        parent_asset=cabinet,
        object_type=ObjectType.BASE,
    )

    assert isinstance(drawer.object_cfg, ArticulationBodyReferenceCfg)
    assert drawer.get_initial_pose().position_xyz == pytest.approx((0.5, 0.0, 0.0))
    # Selecting the variants leaves the shared source layer, and so every stage using it, unchanged.
    assert not source_stage.GetRootLayer().dirty
    assert not source_stage.GetPrimAtPath("/Asset/Cabinet/Drawer").HasAPI(UsdPhysics.RigidBodyAPI)
    return True


def test_articulation_link_reference_selects_the_parent_spawn_variants(tmp_path: pathlib.Path):
    """A link whose rigid body sits behind the parent's spawn variants is posed as a link, without editing its USD."""
    assert run_function_with_persistent_simulation_app(
        _test_articulation_link_reference_selects_the_parent_spawn_variants,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def _test_articulation_link_reference_matches_simulated_link(simulation_app) -> bool:
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    # The microwave door opens toward -pi/2 radians.
    door_angle = -0.5
    microwave = AssetRegistry().get_asset_by_name("microwave")(initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0)))
    microwave.object_cfg.init_state.joint_pos = {"microjoint": door_angle}
    door = ObjectReference(name="door", prim_path="{ENV_REGEX_NS}/microwave/Microwave039_door", parent_asset=microwave)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2"])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(name="microwave_door", scene=Scene(assets=[microwave, door]))
    env = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        env.reset()
        world = env.unwrapped.arena_world
        articulation = env.unwrapped.scene[microwave.name]
        door_joint_index = articulation.data.joint_names.index("microjoint")
        articulation.write_joint_position_to_sim_index(
            position=torch.full((2, 1), door_angle, device=env.unwrapped.device), joint_ids=[door_joint_index]
        )

        T_W_O = world.get_pose_w(door.name)
        torch.testing.assert_close(T_W_O, world.get_body_pose_w(microwave.name, "Microwave039_door"))
        # The door's placement pose is where the simulation puts the door at its initial joint angle.
        T_E_O_expected = door.get_initial_pose().to_tensor(device=T_W_O.device).expand(2, -1)
        T_E_O = world.get_pose_e(door.name)
        torch.testing.assert_close(T_E_O[:, :3], T_E_O_expected[:, :3], atol=1e-3, rtol=0)
        quaternion_alignment = torch.abs(torch.sum(T_E_O[:, 3:] * T_E_O_expected[:, 3:], dim=-1))
        torch.testing.assert_close(quaternion_alignment, torch.ones_like(quaternion_alignment), atol=1e-4, rtol=0)
    finally:
        env.close()
    return True


def test_articulation_link_reference_matches_simulated_link():
    """A door reference reads the simulated door link, and its placement pose matches the initial joint angle."""
    assert run_function_with_persistent_simulation_app(
        _test_articulation_link_reference_matches_simulated_link,
        headless=HEADLESS,
    )


def _test_prismatic_link_placement_and_posed_bounds_match_simulation_at_spawn_scale(
    simulation_app, tmp_path: pathlib.Path
) -> bool:
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd_at_joint_pos

    # The cabinet is authored at scale 2 and spawned at scale 2, so the drawer joint's body0 is scaled by 4.
    spawn_scale = 2.0
    drawer_travel = 0.3
    drawer = _drawer_reference(tmp_path, "prismatic", drawer_travel, spawn_scale)
    cabinet = drawer.parent_asset
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2"])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(name="scaled_drawer", scene=Scene(assets=[cabinet, drawer]))
    env = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        env.reset()
        articulation = env.unwrapped.scene[cabinet.name]
        drawer_joint_index = articulation.data.joint_names.index("drawer_joint")
        # Env 0 closes the drawer and env 1 opens it.
        articulation.write_joint_position_to_sim_index(
            position=torch.tensor([[0.0], [drawer_travel]], device=env.unwrapped.device),
            joint_ids=[drawer_joint_index],
        )
        # The cabinet's base B is free and drops during reset, so measure drawer D from it. B is not rotated.
        world = env.unwrapped.arena_world
        t_W_D = world.get_pose_w(drawer.name)[:, :3]
        t_W_B = world.get_body_pose_w(cabinet.name, "Base")[:, :3]
        t_B_D = (t_W_D - t_W_B).cpu()
    finally:
        env.close()

    # PhysX moves the drawer by its joint position in metres, whatever the scale.
    torch.testing.assert_close(t_B_D[1] - t_B_D[0], torch.tensor([drawer_travel, 0.0, 0.0]), atol=1e-4, rtol=0)
    # The reference is placed where PhysX puts the opened drawer; the base sits at the cabinet's origin.
    torch.testing.assert_close(t_B_D[1], torch.tensor(drawer.get_initial_pose().position_xyz), atol=1e-4, rtol=0)
    # The posed bounds of the drawer move by the same distance.
    scale = (spawn_scale, spawn_scale, spawn_scale)
    closed_bounds = compute_local_bounding_box_from_usd_at_joint_pos(
        cabinet.usd_path, {"drawer_joint": 0.0}, scale, prim_path="/Asset/Cabinet/Drawer"
    )
    opened_bounds = compute_local_bounding_box_from_usd_at_joint_pos(
        cabinet.usd_path, {"drawer_joint": drawer_travel}, scale, prim_path="/Asset/Cabinet/Drawer"
    )
    torch.testing.assert_close(
        opened_bounds.min_point - closed_bounds.min_point, torch.tensor([[drawer_travel, 0.0, 0.0]]), atol=1e-6, rtol=0
    )
    return True


def test_prismatic_link_placement_and_posed_bounds_match_simulation_at_spawn_scale(tmp_path: pathlib.Path):
    """A scaled drawer's reference and posed bounds open by the distance PhysX opens the drawer."""
    assert run_function_with_persistent_simulation_app(
        _test_prismatic_link_placement_and_posed_bounds_match_simulation_at_spawn_scale,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def _kitchen_with_counter(tmp_path: pathlib.Path, spawn_cfg_addon: dict | None = None):
    """Return a kitchen whose default prim is /World, with a counter at (1, 2, 0) and a reference to it.

    The default prim references a prototype prim, as many asset files' default prims reference others. The counter's
    top is a cube of size 0.2 in the selected "small" variant of a "Size" variant set and of size 1 in its "large"
    variant.
    """
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType

    usd_path = tmp_path / "kitchen.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/Prototype/shelf")
    root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddInternalReference("/Prototype")
    UsdGeom.Xform.Define(stage, "/World/counter").AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 0.0))
    size_variants = root.GetVariantSets().AddVariantSet("Size")
    for variant_name, size in (("small", 0.2), ("large", 1.0)):
        size_variants.AddVariant(variant_name)
        size_variants.SetVariantSelection(variant_name)
        with size_variants.GetVariantEditContext():
            top = UsdGeom.Cube.Define(stage, "/World/counter/top")
            top.CreateSizeAttr(size)
            top.CreateExtentAttr([Gf.Vec3f(-size / 2), Gf.Vec3f(size / 2)])
    size_variants.SetVariantSelection("small")
    stage.GetRootLayer().Save()
    kitchen = Object(
        name="kitchen", usd_path=str(usd_path), object_type=ObjectType.BASE, spawn_cfg_addon=spawn_cfg_addon
    )
    return ObjectReference(name="counter", prim_path="{ENV_REGEX_NS}/kitchen/counter", parent_asset=kitchen)


def _test_object_reference_keeps_its_path_in_the_parent_usd_file(simulation_app, tmp_path: pathlib.Path) -> bool:
    counter = _kitchen_with_counter(tmp_path)

    # Placement leaves the counter out of the kitchen's collision mesh by this path, which is read from the USD file.
    assert counter.prim_path_in_parent_usd == "/World/counter"
    assert counter.get_initial_pose().position_xyz == pytest.approx((1.0, 2.0, 0.0))
    return True


def test_object_reference_keeps_its_path_in_the_parent_usd_file(tmp_path: pathlib.Path):
    """A reference records its prim's path in the parent's USD file, not in the stage it inspects."""
    assert run_function_with_persistent_simulation_app(
        _test_object_reference_keeps_its_path_in_the_parent_usd_file,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def _test_object_reference_bounds_follow_the_parent_spawn_variants(simulation_app, tmp_path: pathlib.Path) -> bool:
    counter = _kitchen_with_counter(tmp_path, spawn_cfg_addon={"variants": {"Size": "large"}})

    torch.testing.assert_close(counter.get_bounding_box().size, torch.tensor([[1.0, 1.0, 1.0]]))
    return True


def test_object_reference_bounds_follow_the_parent_spawn_variants(tmp_path: pathlib.Path):
    """A reference's bounds come from the geometry of the variants its parent spawns with."""
    assert run_function_with_persistent_simulation_app(
        _test_object_reference_bounds_follow_the_parent_spawn_variants,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )


def test_object_reference_get_collision_mesh_extracts_referenced_prim(monkeypatch):
    """ObjectReference collision meshes are extracted from the referenced sub-prim."""
    import trimesh

    from isaaclab_arena.assets.object_reference import ObjectReference

    expected_mesh = trimesh.creation.box(extents=(0.2, 0.1, 0.05))
    calls = {"open_count": 0}

    class Stage:
        def GetPrimAtPath(self, prim_path):
            return object()

    def open_spawned_usd_stage():
        calls["open_count"] += 1
        return Stage()

    obj_ref = ObjectReference.__new__(ObjectReference)
    obj_ref.parent_asset = SimpleNamespace(
        usd_path="/tmp/kitchen.usd", name="kitchen", open_spawned_usd_stage=open_spawned_usd_stage
    )
    obj_ref.prim_path = "{ENV_REGEX_NS}/kitchen/counter"
    obj_ref._parent_scale = (2.0, 1.0, 1.0)
    obj_ref._collision_mesh = None
    obj_ref._collision_mesh_loaded = False

    monkeypatch.setattr(
        ObjectReference,
        "isaaclab_prim_path_to_original_prim_path",
        staticmethod(lambda prim_path, parent, stage: "/World/counter"),
    )

    def fake_extract(stage, prim_path, scale):
        calls["extract"] = (prim_path, scale)
        return expected_mesh

    monkeypatch.setattr("isaaclab_arena.assets.object_reference.extract_trimesh_from_prim", fake_extract)

    assert obj_ref.get_collision_mesh() is expected_mesh
    assert obj_ref.get_collision_mesh() is expected_mesh
    assert calls == {
        "open_count": 1,
        "extract": ("/World/counter", (2.0, 1.0, 1.0)),
    }


def test_object_reference_get_collision_mesh_returns_none_on_extraction_failure(monkeypatch):
    """Meshless references fall back to AABB collision instead of aborting aggregation."""
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.utils.usd.helpers import NoCollisionMeshError

    calls = {"extract_count": 0}

    class Stage:
        def GetPrimAtPath(self, prim_path):
            return object()

    obj_ref = ObjectReference.__new__(ObjectReference)
    obj_ref.name = "counter"
    obj_ref.parent_asset = SimpleNamespace(usd_path="/tmp/kitchen.usd", name="kitchen", open_spawned_usd_stage=Stage)
    obj_ref.prim_path = "{ENV_REGEX_NS}/kitchen/counter"
    obj_ref._parent_scale = (1.0, 1.0, 1.0)
    obj_ref._collision_mesh = None
    obj_ref._collision_mesh_loaded = False

    monkeypatch.setattr(
        ObjectReference,
        "isaaclab_prim_path_to_original_prim_path",
        staticmethod(lambda prim_path, parent, stage: "/World/counter"),
    )

    def fail_extract(stage, prim_path, scale):
        calls["extract_count"] += 1
        raise NoCollisionMeshError("No mesh geometry found under /World/counter")

    monkeypatch.setattr("isaaclab_arena.assets.object_reference.extract_trimesh_from_prim", fail_extract)

    assert obj_ref.get_collision_mesh() is None
    assert obj_ref.get_collision_mesh() is None
    assert calls["extract_count"] == 1


def test_object_reference_get_collision_mesh_returns_none_on_unsupported_geometry(monkeypatch):
    """Unsupported reference geometry falls back to AABB collision."""
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.utils.usd.helpers import UnsupportedCollisionGeometryError

    class Stage:
        def GetPrimAtPath(self, prim_path):
            return object()

    obj_ref = ObjectReference.__new__(ObjectReference)
    obj_ref.name = "counter"
    obj_ref.parent_asset = SimpleNamespace(usd_path="/tmp/kitchen.usd", name="kitchen", open_spawned_usd_stage=Stage)
    obj_ref.prim_path = "{ENV_REGEX_NS}/kitchen/counter"
    obj_ref._parent_scale = (1.0, 1.0, 1.0)
    obj_ref._collision_mesh = None
    obj_ref._collision_mesh_loaded = False

    monkeypatch.setattr(
        ObjectReference,
        "isaaclab_prim_path_to_original_prim_path",
        staticmethod(lambda prim_path, parent, stage: "/World/counter"),
    )

    def fail_extract(stage, prim_path, scale):
        raise UnsupportedCollisionGeometryError("Unsupported non-mesh geometry under /World/counter: /World/cube")

    monkeypatch.setattr("isaaclab_arena.assets.object_reference.extract_trimesh_from_prim", fail_extract)

    assert obj_ref.get_collision_mesh() is None


def test_object_reference_get_collision_mesh_raises_on_missing_prim(monkeypatch):
    """Bad reference paths are configuration errors, not meshless fallback cases."""
    import pytest

    from isaaclab_arena.assets.object_reference import ObjectReference

    class Stage:
        def GetPrimAtPath(self, prim_path):
            return None

    obj_ref = ObjectReference.__new__(ObjectReference)
    obj_ref.name = "counter"
    obj_ref.parent_asset = SimpleNamespace(usd_path="/tmp/kitchen.usd", name="kitchen", open_spawned_usd_stage=Stage)
    obj_ref.prim_path = "{ENV_REGEX_NS}/kitchen/missing"
    obj_ref._parent_scale = (1.0, 1.0, 1.0)
    obj_ref._collision_mesh = None
    obj_ref._collision_mesh_loaded = False

    monkeypatch.setattr(
        ObjectReference,
        "isaaclab_prim_path_to_original_prim_path",
        staticmethod(lambda prim_path, parent, stage: "/World/missing"),
    )

    with pytest.raises(ValueError, match="No prim found"):
        obj_ref.get_collision_mesh()


def test_object_reference_world_bbox_without_parent_pose_uses_reference_pose():
    """A reference without a parent pose is placed directly from its relative prim pose."""
    obj_ref = _object_reference_with_cached_bbox(
        parent_pose=None,
        relative_pose=Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        bbox=AxisAlignedBoundingBox(min_point=(-0.1, -0.2, 0.0), max_point=(0.1, 0.2, 0.3)),
    )

    world_bbox = obj_ref.get_world_bounding_box()

    assert torch.allclose(world_bbox.min_point, torch.tensor([[0.9, 1.8, 3.0]]), atol=1e-6)
    assert torch.allclose(world_bbox.max_point, torch.tensor([[1.1, 2.2, 3.3]]), atol=1e-6)


def get_test_scene():
    from isaaclab_arena.assets.registries import AssetRegistry  # noqa: F401
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()

    kitchen = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    kitchen.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(3.69020713150969, -0.804121657812894, 1.2531903565606817), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)
        )
    )
    microwave.set_initial_pose(
        Pose(
            position_xyz=(2.862758610786719, -0.39786255771393336, 1.087924015237011),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )

    return Scene(assets=[kitchen, cracker_box, microwave])


def _test_reference_objects_with_background_pose(background_pose: Pose, tmp_path: pathlib.Path) -> bool:

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tests.utils.pick_and_place import lift_settled_objects_once

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    # Get the test scene
    scene = get_test_scene()
    print(f"Saving a test USD to {tmp_path}")
    scene.export_to_usd(tmp_path)

    # Scene
    # Contains 3 reference objects:
    # - cracker box (target object)
    # - drawer (destination location)
    # - microwave (openable object)
    background = background_from_usd_path(name="kitchen", usd_path=tmp_path, initial_pose=background_pose)
    embodiment = FrankaIKEmbodiment()
    cracker_box = ObjectReference(
        name="cracker_box",
        prim_path="{ENV_REGEX_NS}/kitchen/cracker_box",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    destination_location = ObjectReference(
        name="drawer",
        prim_path="{ENV_REGEX_NS}/kitchen/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    microwave = OpenableObjectReference(
        name="microwave",
        prim_path="{ENV_REGEX_NS}/kitchen/microwave",
        parent_asset=background,
        openable_joint_name="microjoint",
        openable_threshold=0.5,
    )

    scene = Scene(assets=[background, cracker_box, destination_location, microwave])

    # Build the environment
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="reference_object_test",
        embodiments=[embodiment],
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    try:

        def open_microwave():
            with torch.inference_mode():
                microwave.open(env, env_ids=None, asset_cfg=SceneEntityCfg(microwave.name))

        def close_microwave():
            with torch.inference_mode():
                microwave.close(env, env_ids=None, asset_cfg=SceneEntityCfg(microwave.name))

        close_microwave()

        # Run some zero actions.
        terminated_list: list[bool] = []
        success_list: list[bool] = []
        open_list: list[bool] = []
        lifted_envs = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                if _ == OPEN_STEP:
                    open_microwave()
                lift_settled_objects_once(env.unwrapped, cracker_box.name, lifted_envs)
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)
                success = env.unwrapped.termination_manager.get_term("success")
                is_open = microwave.is_open(env, SceneEntityCfg(microwave.name))
                terminated_list.append(terminated.item())
                success_list.append(success.item())
                open_list.append(is_open.item())

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    # Check that the termination condition is:
    # - not met at the start (object above the drawer)
    # - met at the end (object in the drawer)
    print("Checking scene started not terminated and then became terminated")
    print(f"terminated_list: {terminated_list}")
    assert np.any(np.array(terminated_list))  # == True
    assert np.any(np.logical_not(np.array(terminated_list)))  # == False
    print("Checking scene started not success and then became success")
    print(f"success_list: {success_list}")
    assert np.any(np.array(success_list))  # == True
    assert np.any(np.logical_not(np.array(success_list)))  # == False
    print("Checking that the microwave started not open and then became open")
    print(f"open_list: {open_list}")
    assert np.any(np.array(open_list))  # == True
    assert np.any(np.logical_not(np.array(open_list)))  # == False

    return True


def _test_reference_objects(simulation_app, tmp_path: pathlib.Path) -> bool:
    return _test_reference_objects_with_background_pose(Pose.identity(), tmp_path)


def _test_reference_objects_with_transform(simulation_app, tmp_path: pathlib.Path) -> bool:
    background_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_xyzw=(0, 0, -0.70711, 0.70711))
    return _test_reference_objects_with_background_pose(background_pose, tmp_path)


def test_reference_objects(tmp_path: pathlib.Path):
    tmp_path = tmp_path / "reference_objects.usd"
    result = run_function_with_persistent_simulation_app(
        _test_reference_objects,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, "Test failed"


def test_reference_objects_with_transform(tmp_path: pathlib.Path):
    # NOTE(alexmillane, 2025-11-25): The idea here is to test that
    # the test still works if the whole environment is translated and rotated.
    # This relies on the reference objects relative poses being correct.
    tmp_path = tmp_path / "reference_objects_with_transform.usd"
    result = run_function_with_persistent_simulation_app(
        _test_reference_objects_with_transform,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_reference_objects()
    test_reference_objects_with_transform()
