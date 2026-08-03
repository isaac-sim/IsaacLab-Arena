# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_embodiment_provides_robot_collision_mesh(simulation_app) -> bool:
    """Check the embodiment exposes its robot mesh so MESH mode does not fall back to the bbox proxy."""

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

    try:
        emb = DroidAbsoluteJointPositionEmbodiment()

        mesh = emb.get_collision_mesh()
        assert mesh is not None, "embodiment must expose a collision mesh; None forces the loose bbox fallback"
        assert len(mesh.vertices) > 0

        # The spawn USD composes robot and stand, so the mesh spans the whole assembly: 1.46 x 0.91 x
        # 2.10 m as measured, the tallest axis being the arm on its 1.35 m stand. A leaked 50 m ground
        # plane would still blow this up by an order of magnitude.
        extents = mesh.extents
        assert all(e < 3.0 for e in extents), f"mesh leaked non-robot geometry: extents {extents}"

        # Link boxes conservatively cover the same posed Gprims as the placement bbox.
        bbox = emb.get_bounding_box()
        bbox_size = (bbox.max_point - bbox.min_point)[0].tolist()
        for mesh_extent, box_extent in zip(extents, bbox_size):
            assert (
                mesh_extent + 1e-3 >= box_extent
            ), f"robot mesh extents {extents.tolist()} should cover placement bbox {bbox_size}"
            assert abs(mesh_extent - box_extent) < 0.2, f"mesh extents {extents} disagree with box {bbox_size}"

        # Cached geometry is copied before return so caller mutation cannot poison later queries.
        another_mesh = emb.get_collision_mesh()
        assert another_mesh is not mesh
        original_vertex = another_mesh.vertices[0].copy()
        mesh.vertices[0] += 100.0
        assert (emb.get_collision_mesh().vertices[0] == original_vertex).all()

        another_bbox = emb.get_bounding_box()
        assert another_bbox is not bbox
        original_min = another_bbox.min_point.clone()
        bbox.min_point.add_(100.0)
        assert emb.get_bounding_box().min_point.equal(original_min)

        # Isaac Lab reaches placeable assets through EventTermCfg params and validates whatever they
        # hold without tracking visited objects, so an embodiment holding its mesh would send
        # validation recursing through trimesh's back-references until the stack overflows.
        from isaaclab.managers import EventTermCfg

        def _noop(env, env_ids, embodiment):
            pass

        EventTermCfg(func=_noop, mode="reset", params={"embodiment": emb}).validate()

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_spawn_pose_matches_the_reset_pose(simulation_app) -> bool:
    """The spawn joint positions placement geometry is posed at are the ones a reset drives to.

    Also covers constructor joint-pose overrides, which must update the spawn state placement
    geometry reads.
    """

    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd_articulation import resolve_joint_pos_patterns

    franka_override = [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]
    droid_override = [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400] + [0.0] * 5

    for embodiment_class, override in (
        (FrankaIKEmbodiment, None),
        (DroidAbsoluteJointPositionEmbodiment, None),
        (FrankaIKEmbodiment, franka_override),
        (DroidAbsoluteJointPositionEmbodiment, droid_override),
    ):
        env = None
        try:
            embodiment = embodiment_class(initial_joint_pose=override)
            environment = IsaacLabArenaEnvironment(
                name=f"spawn_pose_{embodiment_class.__name__}_{override is not None}",
                embodiment=embodiment,
                scene=Scene(assets=[]),
            )
            builder_cfg = arena_env_builder_cfg_from_argparse(get_isaaclab_arena_cli_parser().parse_args([]))
            env = ArenaEnvBuilder(environment, builder_cfg).make_registered()
            env.reset()

            robot = env.unwrapped.scene["robot"]
            joint_names = list(robot.joint_names)
            spawn_pose = resolve_joint_pos_patterns(joint_names, embodiment.get_placement_geometry_source().joint_pos)
            # Both sides below are read back from the spawn state, so pin the override against what
            # was asked for; otherwise a setter that quietly dropped it would still agree with itself.
            if override is not None:
                expected_override = dict(zip(joint_names, override))
                assert spawn_pose == expected_override, (
                    "set_initial_joint_pose did not reach the spawn state placement geometry reads:"
                    f" {spawn_pose} != {expected_override}"
                )
            # Compare against the defaults a reset restores rather than measured positions, which
            # carry gravity sag and the joint randomisation event on top.
            reset_to = torch.as_tensor(robot.data.default_joint_pos)[0].cpu()
            for index, joint_name in enumerate(joint_names):
                assert abs(spawn_pose.get(joint_name, 0.0) - float(reset_to[index])) < 1e-6, (
                    f"{embodiment_class.__name__}"
                    f"{' with an overridden pose' if override is not None else ''} poses placement geometry with"
                    f" {joint_name} at {spawn_pose.get(joint_name, 0.0)}, but resets it to {float(reset_to[index])}"
                )

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return False

        finally:
            if env is not None:
                env.close()

    return True


def _test_joint_position_action_offset_is_pinned(simulation_app) -> bool:
    """Reposing the arm leaves the joint-position action's zero point where policies were fitted.

    The zero point is a displacement origin trained policies depend on, so it is spelled out here
    rather than read from the config: this fails if the spawn pose starts feeding it again, or if an
    Isaac Lab bump moves the default pose it is derived from.
    """
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment

    trained_against = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
    }

    try:
        embodiment = FrankaJointPosEmbodiment()
        embodiment.set_initial_joint_pose([0.0] * 9)
        arm_action = embodiment.action_config.arm_action
        assert not arm_action.use_default_offset, "the action zero point would follow the spawn pose"
        assert arm_action.offset == trained_against, f"action zero point moved to {arm_action.offset}"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_embodiment_provides_robot_collision_mesh():
    """Pytest entry point for the embodiment collision-mesh test."""
    result = run_function_with_persistent_simulation_app(_test_embodiment_provides_robot_collision_mesh, headless=True)
    assert result, f"Test {test_embodiment_provides_robot_collision_mesh.__name__} failed"


def test_spawn_pose_matches_the_reset_pose():
    """Pytest entry point for the spawn-versus-reset joint-position test."""
    result = run_function_with_persistent_simulation_app(_test_spawn_pose_matches_the_reset_pose, headless=True)
    assert result, f"Test {test_spawn_pose_matches_the_reset_pose.__name__} failed"


def test_joint_position_action_offset_is_pinned():
    """Pytest entry point for the joint-position action offset test."""
    result = run_function_with_persistent_simulation_app(_test_joint_position_action_offset_is_pinned, headless=True)
    assert result, f"Test {test_joint_position_action_offset_is_pinned.__name__} failed"


if __name__ == "__main__":
    test_embodiment_provides_robot_collision_mesh()
    test_spawn_pose_matches_the_reset_pose()
    test_joint_position_action_offset_is_pinned()
