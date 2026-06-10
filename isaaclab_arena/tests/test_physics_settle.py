# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim tests for the physics-settle primitives and the pooled placement-validation sweep.

Each scenario runs end to end against a live SimulationApp:
- objects at rest stay at rest -> objects_settled_per_episode() is True,
- two objects launched on a collision course keep moving -> objects_settled_per_episode() is False,
- a per-env batch of layouts is graded independently in one settle pass,
- validate_pool_layouts() sweeps every pooled candidate and stamps each layout's PHYSICS_SETTLED verdict.
"""

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
SETTLE_STEPS = 30
LIN_VEL_THRESH = 0.1
ANG_VEL_THRESH = 0.1


def _floating_sphere_cfg():
    """A gravity-free sphere so motion (or its absence) is driven only by initial velocity and contacts."""
    import isaaclab.sim as sim_utils

    return sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
    )


def _build_two_sphere_env(poses, velocities, num_envs=1):
    """Build a scene with two named spheres at the given poses and initial velocities.

    ``poses`` may be plain ``Pose`` objects (same layout in every env) or ``PosePerEnv`` objects
    (distinct layout per env), which is how the parallel layout-validation case diverges the envs.

    Returns the gym-wrapped env (already reset) and the two object names.
    """
    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    spheres = []
    for instance_name, pose, velocity in zip(("sphere_a", "sphere_b"), poses, velocities):
        sphere = Sphere(instance_name=instance_name, spawner_cfg=_floating_sphere_cfg())
        sphere.set_initial_pose(pose)
        sphere.set_initial_velocity(velocity)
        spheres.append(sphere)

    scene = Scene(assets=spheres)
    isaaclab_arena_environment = IsaacLabArenaEnvironment(name="physics_settle_test", scene=scene)

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = num_envs
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()
    return env, [sphere.name for sphere in spheres]


def _test_objects_settled_when_at_rest(simulation_app):
    """Two well-separated, motionless spheres should report settled after stepping physics."""
    from isaaclab_arena.utils import physics_settle
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.velocity import Velocity

    poses = [
        Pose(position_xyz=(-0.5, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        Pose(position_xyz=(0.5, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    ]
    velocities = [Velocity.zero(), Velocity.zero()]

    env, object_names = _build_two_sphere_env(poses, velocities)
    try:
        physics_settle.step_physics(env, SETTLE_STEPS)
        settled = physics_settle.objects_settled_per_episode(env, [0], object_names, LIN_VEL_THRESH, ANG_VEL_THRESH)[0]
        assert settled, "Motionless objects should settle"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_objects_not_settled_when_colliding(simulation_app):
    """Two spheres launched at each other keep moving, so the settle check should report unsettled."""
    from isaaclab_arena.utils import physics_settle
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.velocity import Velocity

    poses = [
        Pose(position_xyz=(-0.3, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        Pose(position_xyz=(0.3, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    ]
    # Head-on along x at well above the settle threshold; with gravity disabled they coast into
    # each other and keep moving through and after the collision.
    velocities = [
        Velocity(linear_xyz=(1.0, 0.0, 0.0)),
        Velocity(linear_xyz=(-1.0, 0.0, 0.0)),
    ]

    env, object_names = _build_two_sphere_env(poses, velocities)
    try:
        physics_settle.step_physics(env, SETTLE_STEPS)
        settled = physics_settle.objects_settled_per_episode(env, [0], object_names, LIN_VEL_THRESH, ANG_VEL_THRESH)[0]
        assert not settled, "Colliding objects should not settle"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _build_parallel_layout_env(num_envs, expected_settled):
    """Build a multi-env scene with a per-env layout: a kinematic floor plus two gravity-driven spheres.

    Each env gets a distinct layout via ``PosePerEnv``. "Good" layouts rest the spheres on the floor
    (they stay put); "bad" layouts drop them from height (they are still falling during the window).
    Returns the gym-wrapped env (already reset) and the two sphere names (the floor is excluded, like a
    background/anchor object would be in a real settle check).
    """
    from isaaclab_arena.assets.object_library import ProceduralTable, Sphere
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    floor_top_z = 0.02  # ProceduralTable is 0.04 thick, centered at z=0.
    rest_z = floor_top_z + 0.1  # Sphere radius is 0.1, so this rests it on the floor.
    drop_z = 1.0  # High enough that the sphere is still falling at the end of the settle window.

    def sphere_poses(x):
        poses = []
        for env_id in range(num_envs):
            z = rest_z if expected_settled[env_id] else drop_z
            poses.append(Pose(position_xyz=(x, 0.0, z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        return PosePerEnv(poses=poses)

    floor = ProceduralTable(instance_name="floor")
    floor.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    sphere_a = Sphere(instance_name="sphere_a")
    sphere_b = Sphere(instance_name="sphere_b")
    sphere_a.set_initial_pose(sphere_poses(-0.2))
    sphere_b.set_initial_pose(sphere_poses(0.2))

    scene = Scene(assets=[floor, sphere_a, sphere_b])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(name="physics_settle_parallel_test", scene=scene)

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = num_envs
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()
    return env, [sphere_a.name, sphere_b.name]


def _test_parallel_layout_validation_per_env(simulation_app):
    """Validate parallel layouts in one pass: each env gets its own layout and is graded independently.

    Odd env ids rest their spheres on the floor (settle); even env ids drop them from height (still
    falling, so unsettled). The settle check must return the matching per-env verdict for every env,
    mirroring how the pool validates a batch of placement layouts across all envs at once.
    """
    from isaaclab_arena.utils import physics_settle

    num_envs = 4
    expected_settled = [env_id % 2 == 1 for env_id in range(num_envs)]

    env, object_names = _build_parallel_layout_env(num_envs, expected_settled)
    try:
        physics_settle.step_physics(env, SETTLE_STEPS)
        settled_per_env = physics_settle.objects_settled_per_episode(
            env, list(range(num_envs)), object_names, LIN_VEL_THRESH, ANG_VEL_THRESH
        )
        print(f"Per-env settle verdicts={settled_per_env} expected={expected_settled}")
        assert (
            settled_per_env == expected_settled
        ), f"Per-env settle verdicts {settled_per_env} != expected {expected_settled}"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_validate_pool_layouts_grades_each_layout(simulation_app):
    """validate_pool_layouts should step physics on every pooled candidate and stamp each layout's
    PHYSICS_SETTLED verdict onto its own checklist.

    Builds a real Arena env (an office table anchor with a cracker box placed On it, from the asset
    registry) across two envs, then seeds each env's pool queue with two candidates: one resting layout
    (settles within the window) and one lifted a couple of meters above the table (still falling, so
    unsettled). The sweep validates all four candidates -- two per env, batched in parallel across the two
    scene envs -- and must record the matching settled/unsettled result on each candidate while leaving the
    geometric (gating) checks untouched.
    """
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_library import CrackerBox, OfficeTable
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.placement_pool_validation import validate_pool_layouts
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationChecklist
    from isaaclab_arena.relations.pooled_object_placer import EnvLayoutPool
    from isaaclab_arena.relations.relations import IsAnchor, On, get_anchor_objects
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    num_envs = 2
    drop_height = 2.0

    table = OfficeTable(instance_name="table")
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    # OfficeTable is a free dynamic rigid body and the env has no ground plane, so without pinning it would
    # fall during stepping and drag the cracker with it (nothing would ever settle). Make the anchor surface
    # kinematic so it holds still, exactly like the kinematic procedural/background tables.
    table.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    cracker = CrackerBox(instance_name="cracker")
    cracker.add_relation(On(table, clearance_m=0.01))

    scene = Scene(assets=[table, cracker])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(name="validate_pool_layouts_test", scene=scene)

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = num_envs
    env = ArenaEnvBuilder(isaaclab_arena_environment, args_cli).make_registered()
    env.reset()

    try:
        pool = get_placement_pool(env)
        assert pool is not None, "Pool validation requires a pooled placer on the env."
        # The settle config drives the sweep; max_retries is unused by validate_pool_layouts.
        # num_steps is in env-step units; validate_pool_layouts converts it to physics substeps via the
        # env's decimation. Target ~60 substeps (~0.3 s at dt=0.005): the resting box settles within the
        # window, while the 2 m drop keeps the lifted box airborne (it falls only ~0.44 m in that span).
        settle_steps = max(1, round(60 / env.unwrapped.cfg.decimation))
        settle_params = PhysicsSettleParams(
            num_steps=settle_steps,
            lin_vel_thresh=LIN_VEL_THRESH,
            ang_vel_thresh=ANG_VEL_THRESH,
        )

        # The builder re-instantiates the scene objects, so resolve the actual cracker box the pool holds
        # (not the local one passed to the scene) as the single movable object.
        anchor_objects_set = set(get_anchor_objects(pool.objects))
        movable = [obj for obj in pool.objects if obj not in anchor_objects_set]
        assert len(movable) == 1, f"Expected exactly one movable object, got {[o.name for o in movable]}."
        cracker_obj = movable[0]

        # Every stored solver layout rests the cracker box on the table (the On relation), so use one of
        # them (env-local, valid in any env) as the resting candidate, and lift a copy by drop_height for
        # the unstable one.
        rest_x, rest_y, rest_z = pool.layouts_per_env()[0][0].positions[cracker_obj]
        unstable_z = rest_z + drop_height

        def _layout(z: float) -> PlacementResult:
            # Each candidate carries its own checklist so the sweep stamps PHYSICS_SETTLED independently.
            return PlacementResult(
                validation_checklist=PlacementValidationChecklist(
                    checklist_items={PlacementCheck.NO_OVERLAP: True, PlacementCheck.ON_RELATION: True},
                    required_items={PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION},
                ),
                positions={cracker_obj: (rest_x, rest_y, z)},
                final_loss=0.0,
                attempts=1,
            )

        # Seed each env's queue with a resting and a dropped candidate, ordered differently per env so the
        # expected verdict varies by both env and candidate index (not a fixed pattern).
        # env 0: [resting -> settles, dropped -> unsettled]; env 1: [dropped -> unsettled, resting -> settles].
        seeded = {
            0: [_layout(rest_z), _layout(unstable_z)],
            1: [_layout(unstable_z), _layout(rest_z)],
        }
        expected_settled = {(0, 0): True, (0, 1): False, (1, 0): False, (1, 1): True}
        for env_id, env_layouts in seeded.items():
            pool._env_pools[env_id] = EnvLayoutPool(env_layouts)

        results = validate_pool_layouts(env, pool, settle_params)

        assert len(results) == len(expected_settled), f"Expected {len(expected_settled)} candidates, got {len(results)}"
        for env_id, candidate_index, checklist in results:
            assert (
                PlacementCheck.PHYSICS_SETTLED in checklist.checklist_items
            ), f"env {env_id} candidate {candidate_index}: settle sweep did not stamp PHYSICS_SETTLED."
            settled = checklist.checklist_items[PlacementCheck.PHYSICS_SETTLED]
            want = expected_settled[(env_id, candidate_index)]
            print(
                f"env {env_id} candidate {candidate_index}: settled={settled} expected={want} -> {checklist.report()}"
            )
            assert settled is want, f"env {env_id} candidate {candidate_index}: settled={settled}, expected {want}"
            # PHYSICS_SETTLED is optional, so the geometric gate must still pass even for the dropped layout.
            assert (
                checklist.pass_validation_checklist()
            ), f"env {env_id} candidate {candidate_index}: geometric gate should be unaffected by the settle sweep."
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def test_objects_settled_when_at_rest():
    result = run_simulation_app_function(_test_objects_settled_when_at_rest, headless=HEADLESS)
    assert result, f"Test {test_objects_settled_when_at_rest.__name__} failed"


def test_objects_not_settled_when_colliding():
    result = run_simulation_app_function(_test_objects_not_settled_when_colliding, headless=HEADLESS)
    assert result, f"Test {test_objects_not_settled_when_colliding.__name__} failed"


def test_parallel_layout_validation_per_env():
    result = run_simulation_app_function(_test_parallel_layout_validation_per_env, headless=HEADLESS)
    assert result, f"Test {test_parallel_layout_validation_per_env.__name__} failed"


def test_validate_pool_layouts_grades_each_layout():
    result = run_simulation_app_function(_test_validate_pool_layouts_grades_each_layout, headless=HEADLESS)
    assert result, f"Test {test_validate_pool_layouts_grades_each_layout.__name__} failed"


if __name__ == "__main__":
    test_objects_settled_when_at_rest()
    test_objects_not_settled_when_colliding()
    test_parallel_layout_validation_per_env()
    test_validate_pool_layouts_grades_each_layout()
