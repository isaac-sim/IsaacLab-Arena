# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that the turntable disc spins at its drive speed and carries an object resting on it."""

import math
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_ENVS = 2
HEADLESS = True
SPIN_UP_TIME_S = 1.0
MEASUREMENT_TIME_S = 3.0
SPIN_SPEED_TOLERANCE = 0.05
CARRIED_ROTATION_TOLERANCE_RAD = math.radians(10.0)
BASE_DRIFT_TOLERANCE = 1e-3
DISC_JOINT_NAME = "disc_joint"


def _make_turntable_environment():
    """Create a Maple table scene with a banana resting on a turntable."""
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(NUM_ENVS)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("maple_table_robolab")()
    turntable = asset_registry.get_asset_by_name("turntable")()
    banana = asset_registry.get_asset_by_name("banana_ycb_robolab")(name="banana")
    embodiment = asset_registry.get_asset_by_name("droid_rel_joint_pos")(enable_cameras=False)

    table_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )
    table_reference.add_relation(IsAnchor())
    turntable.add_relation(On(table_reference))
    banana.add_relation(On(turntable))

    scene = Scene(assets=[background, table_reference, turntable, banana])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="turntable_banana",
        embodiment=embodiment,
        scene=scene,
        task=NoTask(),
    )
    env = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    return env, turntable, banana


def _read_disc_drive_target_speed_rad_s(usd_path: str) -> float:
    """Return the target velocity of the disc joint's angular drive authored in the turntable USD."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    disc_joint_prims = [prim for prim in stage.Traverse() if prim.GetName() == DISC_JOINT_NAME]
    assert len(disc_joint_prims) == 1, f"Expected one '{DISC_JOINT_NAME}' prim in {usd_path}"
    disc_drive = UsdPhysics.DriveAPI(disc_joint_prims[0], UsdPhysics.Tokens.angular)
    return math.radians(disc_drive.GetTargetVelocityAttr().Get())


def _step_for(env, duration_s: float) -> None:
    import torch

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    num_steps = math.ceil(duration_s / env.unwrapped.step_dt)
    for _ in range(num_steps):
        with torch.inference_mode():
            env.step(actions)


def _test_turntable_carries_object(simulation_app) -> bool:
    import torch

    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    from isaaclab_arena.utils.joint_utils import get_joint_position_from_articulation

    env, turntable, banana = _make_turntable_environment()
    target_speed_rad_s = _read_disc_drive_target_speed_rad_s(turntable.usd_path)
    # The turntable's bounding box is centred on its axis. Its top is the disc's top face plus the 0.5 mm marker stripe.
    turntable_bounding_box = turntable.get_bounding_box()
    disc_radius_m = turntable_bounding_box.max_point[0, 0].item()
    disc_top_height_m = turntable_bounding_box.max_point[0, 2].item()

    try:
        env.reset()
        articulation = env.unwrapped.scene.articulations[turntable.name]
        arena_world = env.unwrapped.arena_world
        disc_joint_index = articulation.data.joint_names.index(DISC_JOINT_NAME)

        _step_for(env, SPIN_UP_TIME_S)
        disc_speed_rad_s = articulation.data.joint_vel.torch[:, disc_joint_index].clone()
        disc_angle_start_rad = get_joint_position_from_articulation(articulation, DISC_JOINT_NAME).clone()
        _, _, banana_yaw_start_rad = euler_xyz_from_quat(arena_world.get_pose_w(banana.name)[:, 3:7])
        T_W_turntable_start = arena_world.get_pose_w(turntable.name).clone()

        _step_for(env, MEASUREMENT_TIME_S)
        disc_angle_end_rad = get_joint_position_from_articulation(articulation, DISC_JOINT_NAME).clone()
        T_W_banana = arena_world.get_pose_w(banana.name).clone()
        T_W_turntable = arena_world.get_pose_w(turntable.name).clone()
        _, _, banana_yaw_end_rad = euler_xyz_from_quat(T_W_banana[:, 3:7])

        disc_rotation_rad = disc_angle_end_rad - disc_angle_start_rad
        banana_rotation_rad = wrap_to_pi(banana_yaw_end_rad - banana_yaw_start_rad)
        carried_rotation_error_rad = wrap_to_pi(banana_rotation_rad - disc_rotation_rad)
        banana_distance_from_axis_m = torch.linalg.vector_norm(T_W_banana[:, :2] - T_W_turntable[:, :2], dim=-1)
        banana_height_above_disc_m = T_W_banana[:, 2] - (T_W_turntable[:, 2] + disc_top_height_m)

        print(f"Disc speed (rad/s): {disc_speed_rad_s.tolist()}, target {target_speed_rad_s:.4f}")
        print(f"Disc rotation (rad): {disc_rotation_rad.tolist()}")
        print(f"Banana rotation (rad): {banana_rotation_rad.tolist()}")
        print(f"Banana distance from the turntable axis (m): {banana_distance_from_axis_m.tolist()}")
        print(f"Banana height above the disc (m): {banana_height_above_disc_m.tolist()}")

        assert torch.allclose(
            disc_speed_rad_s,
            torch.full_like(disc_speed_rad_s, target_speed_rad_s),
            rtol=SPIN_SPEED_TOLERANCE,
        ), "The disc did not reach its drive speed"
        expected_rotation_rad = target_speed_rad_s * MEASUREMENT_TIME_S
        assert torch.allclose(
            disc_rotation_rad,
            torch.full_like(disc_rotation_rad, expected_rotation_rad),
            rtol=SPIN_SPEED_TOLERANCE,
        ), "The disc did not turn by speed times time"
        assert torch.allclose(
            T_W_turntable, T_W_turntable_start, atol=BASE_DRIFT_TOLERANCE
        ), "The turntable base moved while the disc spun"
        assert (
            torch.abs(carried_rotation_error_rad) < CARRIED_ROTATION_TOLERANCE_RAD
        ).all(), "The banana did not turn with the disc"
        assert (banana_distance_from_axis_m < disc_radius_m).all(), "The banana left the disc"
        assert (banana_height_above_disc_m > 0.0).all(), "The banana is below the disc's top face"

        # A reset restores the default joint state; the drive then spins the disc up again.
        env.reset()
        _step_for(env, SPIN_UP_TIME_S)
        disc_speed_after_reset_rad_s = articulation.data.joint_vel.torch[:, disc_joint_index].clone()
        print(f"Disc speed after reset (rad/s): {disc_speed_after_reset_rad_s.tolist()}")
        assert torch.allclose(
            disc_speed_after_reset_rad_s,
            torch.full_like(disc_speed_after_reset_rad_s, target_speed_rad_s),
            rtol=SPIN_SPEED_TOLERANCE,
        ), "The disc did not spin up again after a reset"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def test_turntable_carries_object():
    result = run_function_with_persistent_simulation_app(_test_turntable_carries_object, headless=HEADLESS)
    assert result, "The turntable did not carry the banana at its drive speed"


if __name__ == "__main__":
    test_turntable_carries_object()
