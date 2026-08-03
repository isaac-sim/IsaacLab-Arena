# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posing a USD articulation from its physics joints.

The fixture arm places the joint at the world origin and the child link one unit along +X, so a
90 degree rotation about Z must swing the child from (1, 0, 0) to (0, 1, 0).
"""

import math
import numpy as np

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _define_box(stage, path: str, half_extent: float = 0.1) -> None:
    """Define a cube mesh centred on its own origin."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    h = half_extent
    mesh.GetPointsAttr().Set([
        Gf.Vec3f(-h, -h, -h),
        Gf.Vec3f(h, -h, -h),
        Gf.Vec3f(h, h, -h),
        Gf.Vec3f(-h, h, -h),
        Gf.Vec3f(-h, -h, h),
        Gf.Vec3f(h, -h, h),
        Gf.Vec3f(h, h, h),
        Gf.Vec3f(-h, h, h),
    ])
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 1, 2, 3,
        4, 5, 6, 7,
        0, 1, 5, 4,
        2, 3, 7, 6,
        0, 3, 7, 4,
        1, 2, 6, 5,
    ])  # fmt: skip


def _build_two_link_arm(joint_type: str = "revolute"):
    """Build a base link at the origin and a child link at +X, joined at the world origin."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())

    base = UsdGeom.Xform.Define(stage, "/root/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    _define_box(stage, "/root/base/box")

    forearm = UsdGeom.Xform.Define(stage, "/root/forearm")
    forearm.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(forearm.GetPrim())
    _define_box(stage, "/root/forearm/box")

    schema = UsdPhysics.RevoluteJoint if joint_type == "revolute" else UsdPhysics.PrismaticJoint
    joint = schema.Define(stage, "/root/elbow")
    joint.CreateBody0Rel().SetTargets(["/root/base"])
    joint.CreateBody1Rel().SetTargets(["/root/forearm"])
    joint.CreateAxisAttr("Z" if joint_type == "revolute" else "X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-1.0, 0.0, 0.0))
    return stage


def _posed_forearm_origin(stage, joint_pos) -> np.ndarray:
    """Return where the forearm box's rest-pose origin lands once joint_pos is applied."""
    from isaaclab_arena.utils.usd_articulation import compute_posed_prim_world_deltas, resolve_prim_world_delta

    deltas = compute_posed_prim_world_deltas(stage, "/root", joint_pos)
    delta = resolve_prim_world_delta("/root/forearm/box", deltas)
    assert delta is not None, "forearm geometry must inherit its link's delta"
    return (np.array([1.0, 0.0, 0.0, 1.0]) @ delta)[:3]


def _test_revolute_joint_swings_child_link(simulation_app) -> bool:
    """A 90 degree revolute rotation about Z moves the child link from +X to +Y."""
    stage = _build_two_link_arm()

    posed = _posed_forearm_origin(stage, {"elbow": math.pi / 2.0})
    np.testing.assert_allclose(posed, [0.0, 1.0, 0.0], atol=1e-9)

    # Half the rotation must land on the 45 degree diagonal, ruling out a quarter-turn constant.
    posed_45 = _posed_forearm_origin(stage, {"elbow": math.pi / 4.0})
    np.testing.assert_allclose(posed_45, [math.sqrt(0.5), math.sqrt(0.5), 0.0], atol=1e-9)
    return True


def _test_zero_joint_position_preserves_authored_pose(simulation_app) -> bool:
    """Posing at zero reproduces the authored transforms, so unposed callers see no change."""
    stage = _build_two_link_arm()

    np.testing.assert_allclose(_posed_forearm_origin(stage, {"elbow": 0.0}), [1.0, 0.0, 0.0], atol=1e-9)
    # Omitted joints default to zero rather than to the authored pose.
    np.testing.assert_allclose(_posed_forearm_origin(stage, {}), [1.0, 0.0, 0.0], atol=1e-9)
    return True


def _test_prismatic_joint_translates_child_link(simulation_app) -> bool:
    """A prismatic joint slides the child link along its axis."""
    stage = _build_two_link_arm(joint_type="prismatic")

    np.testing.assert_allclose(_posed_forearm_origin(stage, {"elbow": 0.25}), [1.25, 0.0, 0.0], atol=1e-9)
    return True


def _test_authored_pose_away_from_joint_zero_is_corrected(simulation_app) -> bool:
    """Geometry follows the joint values, not the pose the asset was authored in.

    The forearm is authored swung 90 degrees out, which no longer agrees with the joint frames.
    Posing at zero must pull it back onto +X instead of trusting the authored transform.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = _build_two_link_arm()
    forearm = UsdGeom.Xform.Define(stage, "/root/forearm")
    forearm.GetPrim().RemoveProperty("xformOp:translate")
    forearm.ClearXformOpOrder()
    forearm.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))
    forearm.AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(forearm.GetPrim())

    from isaaclab_arena.utils.usd_articulation import compute_posed_prim_world_deltas, resolve_prim_world_delta

    deltas = compute_posed_prim_world_deltas(stage, "/root", {"elbow": 0.0})
    delta = resolve_prim_world_delta("/root/forearm/box", deltas)
    # The authored origin sits at (0, 1, 0); zeroing the joint returns it to (1, 0, 0).
    posed = (np.array([0.0, 1.0, 0.0, 1.0]) @ delta)[:3]
    np.testing.assert_allclose(posed, [1.0, 0.0, 0.0], atol=1e-9)
    return True


def _test_unknown_joint_name_is_rejected(simulation_app) -> bool:
    """A joint position naming a joint the articulation lacks is a configuration error."""
    from isaaclab_arena.utils.usd_articulation import compute_posed_prim_world_deltas

    stage = _build_two_link_arm()
    try:
        compute_posed_prim_world_deltas(stage, "/root", {"shoulder": 0.5})
    except AssertionError as error:
        assert "shoulder" in str(error), f"assertion should name the unknown joint, got: {error}"
        return True
    raise AssertionError("expected an assertion for an unknown joint name")


def _test_joint_pos_patterns_expand_to_joint_names(simulation_app) -> bool:
    """Isaac Lab regex joint keys expand to every joint they full-match, and misses are dropped."""
    from isaaclab_arena.utils.usd_articulation import resolve_joint_pos_patterns

    names = ["panda_joint1", "panda_joint2", "right_outer_knuckle_joint", "left_inner_finger_joint"]
    resolved = resolve_joint_pos_patterns(names, {"panda_joint1": 0.5, "right_outer.*": 0.25})
    assert resolved == {"panda_joint1": 0.5, "right_outer_knuckle_joint": 0.25}, resolved

    # A pattern matching nothing is dropped, leaving those joints at zero rather than raising.
    assert resolve_joint_pos_patterns(names, {"head_.*": 1.0}) == {}

    # A partial match must not count: Isaac Lab full-matches joint names.
    assert resolve_joint_pos_patterns(names, {"panda": 1.0}) == {}

    # Later keys win where patterns overlap, matching Isaac Lab's ordering.
    overlapped = resolve_joint_pos_patterns(names, {"panda_.*": 1.0, "panda_joint2": 2.0})
    assert overlapped == {"panda_joint1": 1.0, "panda_joint2": 2.0}, overlapped
    return True


def _test_droid_geometry_tracks_configured_joint_positions(simulation_app) -> bool:
    """Droid's configured and zero joint positions produce different conservative link-box proxies."""
    from pxr import Usd

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.usd_articulation import articulation_joint_prims
    from isaaclab_arena.utils.usd_helpers import (
        compute_local_bounding_box_from_usd_at_joint_pos,
        extract_trimesh_from_usd_at_joint_pos,
    )

    robot = DroidAbsoluteJointPositionEmbodiment().scene_config.robot
    usd_path = robot.spawn.usd_path

    stage = Usd.Stage.Open(usd_path)
    joint_names = set(articulation_joint_prims(stage.GetDefaultPrim()))
    assert "panda_joint1" in joint_names, f"expected Franka arm joints, found {sorted(joint_names)}"

    # Passed verbatim, so the config's regex keys (e.g. "right_outer.*") must resolve to real joints.
    configured = robot.init_state.joint_pos
    posed = extract_trimesh_from_usd_at_joint_pos(usd_path, configured)
    posed_bbox = compute_local_bounding_box_from_usd_at_joint_pos(usd_path, configured)
    np.testing.assert_allclose(posed.extents, posed_bbox.size.numpy()[0], atol=2e-3)

    # At zero the arm stands straight up, so it is taller and narrower than the configured pose.
    zero = extract_trimesh_from_usd_at_joint_pos(usd_path, {})
    assert zero.extents[2] > posed.extents[2] + 0.2, f"zero pose should stand taller: {zero.extents}"
    assert zero.extents[0] < posed.extents[0] - 0.2, f"zero pose should be narrower: {zero.extents}"
    return True


def _test_droid_posed_bounding_box_covers_all_geometry(simulation_app) -> bool:
    """The link-box proxy covers every posed Gprim, including analytic geometry."""
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.usd_helpers import (
        compute_local_bounding_box_from_usd,
        compute_local_bounding_box_from_usd_at_joint_pos,
        extract_trimesh_from_usd_at_joint_pos,
    )

    robot = DroidAbsoluteJointPositionEmbodiment().scene_config.robot
    usd_path = robot.spawn.usd_path
    configured = robot.init_state.joint_pos

    authored = compute_local_bounding_box_from_usd(usd_path)
    posed = compute_local_bounding_box_from_usd_at_joint_pos(usd_path, configured)
    np.testing.assert_allclose(posed.size.numpy()[0], authored.size.numpy()[0], atol=2e-3)
    np.testing.assert_allclose(posed.min_point.numpy()[0], authored.min_point.numpy()[0], atol=2e-3)

    # Both paths include every Gprim. Link-local boxes are conservative, so their aggregate can be
    # larger than the exact posed bound but must never be smaller.
    mesh_extents = extract_trimesh_from_usd_at_joint_pos(usd_path, configured).extents
    posed_size = posed.size.numpy()[0]
    assert np.all(mesh_extents >= posed_size - 1e-6), f"proxy {mesh_extents} does not cover bbox {posed_size}"
    assert np.all(mesh_extents - posed_size < 0.2), f"link-local proxy is unexpectedly loose: {mesh_extents}"
    return True


def _test_offline_posing_matches_physx_link_poses(simulation_app) -> bool:
    """Offline posing of the real Droid agrees with PhysX's own kinematics for every link.

    This is the ground-truth check: the simulator resolves the articulation independently, so
    matching its link poses rules out frame, axis, and joint-ordering errors that a synthetic
    two-link arm cannot expose. The joints PhysX settled at are used as the input, because a reset
    lets the position-controlled arm sag a few milliradians off the configured values.
    """
    import warp as wp
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd_articulation import (
        articulation_joint_prims,
        compute_posed_prim_world_deltas,
        resolve_joint_pos_patterns,
    )

    embodiment = DroidAbsoluteJointPositionEmbodiment()
    usd_path = embodiment.scene_config.robot.spawn.usd_path

    arena_env = IsaacLabArenaEnvironment(name="verify_articulation", embodiment=embodiment, scene=Scene(assets=[]))
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        body_names = list(robot.body_names)
        sim_pose = wp.to_torch(robot.data.body_link_pose_w)[0].double().cpu().numpy()
        sim_joint_pos = wp.to_torch(robot.data.joint_pos)[0].double().cpu().numpy()
        actual_joint_pos = {name: float(value) for name, value in zip(robot.joint_names, sim_joint_pos)}
    finally:
        env.close()

    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetDefaultPrim()
    resolved = resolve_joint_pos_patterns(articulation_joint_prims(default_prim), actual_joint_pos)
    deltas = compute_posed_prim_world_deltas(stage, default_prim.GetPath().pathString, resolved)

    # Gripper links nest under panda_link8, so resolve body prims by traversal rather than by layout.
    body_prim_paths = {
        prim.GetName(): prim.GetPath().pathString
        for prim in Usd.PrimRange(default_prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }

    def offline_world_position(prim_path: str) -> np.ndarray:
        rest = np.array(
            UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        )
        posed = rest @ deltas[prim_path] if prim_path in deltas else rest
        return posed[3, :3]

    # Compare relative to the root link, which cancels the env origin and the robot's base placement.
    offline_root = offline_world_position(body_prim_paths[body_names[0]])
    for index, name in enumerate(body_names):
        assert name in body_prim_paths, f"no rigid-body prim named {name} under {default_prim.GetPath()}"
        sim_rel = sim_pose[index, :3] - sim_pose[0, :3]
        offline_rel = offline_world_position(body_prim_paths[name]) - offline_root
        np.testing.assert_allclose(offline_rel, sim_rel, atol=1e-5, err_msg=f"link {name} disagrees with PhysX")
    return True


def _test_closed_loop_articulation_poses_a_spanning_tree(simulation_app) -> bool:
    """A redundant second joint to a body is dropped, leaving the spanning-tree pose untouched."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = _build_two_link_arm()
    strut = UsdGeom.Xform.Define(stage, "/root/strut")
    strut.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(strut.GetPrim())
    _define_box(stage, "/root/strut/box")

    shoulder = UsdPhysics.RevoluteJoint.Define(stage, "/root/shoulder")
    shoulder.CreateBody0Rel().SetTargets(["/root/base"])
    shoulder.CreateBody1Rel().SetTargets(["/root/strut"])
    shoulder.CreateAxisAttr("Z")

    # Closes the loop: the forearm is now reachable both directly and via the strut.
    brace = UsdPhysics.RevoluteJoint.Define(stage, "/root/brace")
    brace.CreateBody0Rel().SetTargets(["/root/strut"])
    brace.CreateBody1Rel().SetTargets(["/root/forearm"])
    brace.CreateAxisAttr("Z")

    # The elbow is declared first, so it keeps the forearm and the brace is the closure that drops.
    np.testing.assert_allclose(_posed_forearm_origin(stage, {"elbow": math.pi / 2.0}), [0.0, 1.0, 0.0], atol=1e-9)
    return True


def _test_instanced_geometry_is_posed_with_its_link(simulation_app) -> bool:
    """Instanced geometry contributes to the posed bounds and mesh, and follows the link it hangs off."""
    import tempfile

    from pxr import Gf, UsdGeom

    from isaaclab_arena.utils.usd_helpers import (
        compute_local_bounding_box_from_usd_at_joint_pos,
        extract_trimesh_from_usd_at_joint_pos,
    )

    stage = _build_two_link_arm()
    # The prototype lives outside the default prim, so only the instance can contribute to the bounds.
    UsdGeom.Xform.Define(stage, "/prototypes/mount")
    _define_box(stage, "/prototypes/mount/box", half_extent=0.2)
    mount = UsdGeom.Xform.Define(stage, "/root/forearm/mount")
    mount.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    mount.GetPrim().GetReferences().AddInternalReference("/prototypes/mount")
    mount.GetPrim().SetInstanceable(True)
    assert mount.GetPrim().IsInstance(), "fixture must exercise a real instance"

    with tempfile.TemporaryDirectory() as tmp_dir:
        usd_path = f"{tmp_dir}/arm_with_instanced_mount.usda"
        stage.Export(usd_path)

        rest_mesh = extract_trimesh_from_usd_at_joint_pos(usd_path, {})
        assert rest_mesh.is_watertight
        try:
            extract_trimesh_from_usd_at_joint_pos(usd_path, {}, scale=(-1.0, 1.0, 1.0))
        except AssertionError as error:
            assert "positive" in str(error)
        else:
            raise AssertionError("negative spawn scale must be rejected before it flips SDF signs")

        # The mount reaches 1.7 along +X: forearm at 1.0, mount offset 0.5, half extent 0.2.
        rest = compute_local_bounding_box_from_usd_at_joint_pos(usd_path, {})
        np.testing.assert_allclose(rest.max_point.numpy()[0][0], 1.7, atol=1e-6)
        np.testing.assert_allclose(rest_mesh.vertices.max(axis=0)[0], 1.7)

        # Swinging the elbow must carry the instance with the forearm rather than leave it behind.
        swung = compute_local_bounding_box_from_usd_at_joint_pos(usd_path, {"elbow": math.pi / 2.0})
        np.testing.assert_allclose(swung.max_point.numpy()[0][1], 1.7, atol=1e-6)
        np.testing.assert_allclose(swung.max_point.numpy()[0][0], 0.2, atol=1e-6)
    return True


def test_revolute_joint_swings_child_link():
    assert run_function_with_persistent_simulation_app(_test_revolute_joint_swings_child_link, headless=HEADLESS)


def test_zero_joint_position_preserves_authored_pose():
    assert run_function_with_persistent_simulation_app(
        _test_zero_joint_position_preserves_authored_pose, headless=HEADLESS
    )


def test_prismatic_joint_translates_child_link():
    assert run_function_with_persistent_simulation_app(_test_prismatic_joint_translates_child_link, headless=HEADLESS)


def test_authored_pose_away_from_joint_zero_is_corrected():
    assert run_function_with_persistent_simulation_app(
        _test_authored_pose_away_from_joint_zero_is_corrected, headless=HEADLESS
    )


def test_unknown_joint_name_is_rejected():
    assert run_function_with_persistent_simulation_app(_test_unknown_joint_name_is_rejected, headless=HEADLESS)


def test_joint_pos_patterns_expand_to_joint_names():
    assert run_function_with_persistent_simulation_app(
        _test_joint_pos_patterns_expand_to_joint_names, headless=HEADLESS
    )


def test_droid_geometry_tracks_configured_joint_positions():
    assert run_function_with_persistent_simulation_app(
        _test_droid_geometry_tracks_configured_joint_positions, headless=HEADLESS
    )


def test_droid_posed_bounding_box_covers_all_geometry():
    assert run_function_with_persistent_simulation_app(
        _test_droid_posed_bounding_box_covers_all_geometry, headless=HEADLESS
    )


def test_offline_posing_matches_physx_link_poses():
    assert run_function_with_persistent_simulation_app(_test_offline_posing_matches_physx_link_poses, headless=HEADLESS)


def test_closed_loop_articulation_poses_a_spanning_tree():
    assert run_function_with_persistent_simulation_app(
        _test_closed_loop_articulation_poses_a_spanning_tree, headless=HEADLESS
    )


def test_instanced_geometry_is_posed_with_its_link():
    assert run_function_with_persistent_simulation_app(
        _test_instanced_geometry_is_posed_with_its_link, headless=HEADLESS
    )
