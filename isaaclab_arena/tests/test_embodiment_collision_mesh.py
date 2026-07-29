# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


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

        # The placement bbox covers the same posed assembly plus the analytic gprims that mesh extraction
        # cannot represent, so it is marginally larger but must not diverge.
        bbox = emb.get_bounding_box()
        bbox_size = (bbox.max_point - bbox.min_point)[0].tolist()
        for mesh_extent, box_extent in zip(extents, bbox_size):
            assert (
                box_extent + 1e-3 >= mesh_extent
            ), f"placement bbox {bbox_size} should cover robot mesh extents {extents.tolist()}"
            assert abs(mesh_extent - box_extent) < 0.2, f"mesh extents {extents} disagree with box {bbox_size}"

        # Both derivations open the USD and pose it, so results are cached rather than recomputed per
        # solve step. The cache is keyed by USD, joint positions and scale, so an identical embodiment
        # shares it.
        assert emb.get_collision_mesh() is mesh
        assert DroidAbsoluteJointPositionEmbodiment().get_collision_mesh() is mesh
        assert emb.get_bounding_box() is bbox
        assert DroidAbsoluteJointPositionEmbodiment().get_bounding_box() is bbox

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

    Both robots also reach their arm pose through an event that assigns joints positionally, which
    cannot be read off the config; the articulation supplies the joint order here instead.
    """

    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd_articulation import resolve_joint_pos_patterns

    for embodiment_class in (FrankaIKEmbodiment, DroidAbsoluteJointPositionEmbodiment):
        env = None
        try:
            embodiment = embodiment_class()
            environment = IsaacLabArenaEnvironment(
                name=f"spawn_pose_{embodiment_class.__name__}", embodiment=embodiment, scene=Scene(assets=[])
            )
            builder_cfg = arena_env_builder_cfg_from_argparse(get_isaaclab_arena_cli_parser().parse_args([]))
            env = ArenaEnvBuilder(environment, builder_cfg).make_registered()
            env.reset()

            robot = env.unwrapped.scene["robot"]
            joint_names = list(robot.joint_names)
            spawn_pose = resolve_joint_pos_patterns(joint_names, embodiment.get_placement_geometry_source().joint_pos)
            # Compare against the defaults a reset restores rather than measured positions, which
            # carry gravity sag and the joint randomisation event on top.
            reset_to = torch.as_tensor(robot.data.default_joint_pos)[0].cpu()
            for index, joint_name in enumerate(joint_names):
                assert abs(spawn_pose.get(joint_name, 0.0) - float(reset_to[index])) < 1e-6, (
                    f"{embodiment_class.__name__} poses placement geometry with {joint_name} at"
                    f" {spawn_pose.get(joint_name, 0.0)}, but resets it to {float(reset_to[index])}"
                )

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return False

        finally:
            if env is not None:
                env.close()

    return True


def test_embodiment_provides_robot_collision_mesh():
    """Pytest entry point for the embodiment collision-mesh test."""
    result = run_simulation_app_function(_test_embodiment_provides_robot_collision_mesh, headless=True)
    assert result, f"Test {test_embodiment_provides_robot_collision_mesh.__name__} failed"


def test_spawn_pose_matches_the_reset_pose():
    """Pytest entry point for the spawn-versus-reset joint-position test."""
    result = run_simulation_app_function(_test_spawn_pose_matches_the_reset_pose, headless=True)
    assert result, f"Test {test_spawn_pose_matches_the_reset_pose.__name__} failed"


if __name__ == "__main__":
    test_embodiment_provides_robot_collision_mesh()
    test_spawn_pose_matches_the_reset_pose()
