# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ``galileo_g1_static_pick_and_place`` (WBC-balanced G1, no nav).

Mirrors ``test_g1_locomanip_apple_to_plate.py`` (same 23-D action layout, same
standing-pose hold actions during warmup) but uses the AGILE WBC backend
(``G1WBCAgileJointEmbodiment``) -- production env defaults to ``g1_wbc_agile_pink``,
so the joint twin keeps the test honest to the deployed stack while still skipping the
PinkIK forward solve for speed.
"""

import torch
import traceback

import pytest
import warp as wp

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
WARMUP_STEPS = 50
# Steps allowed for the once-teleported apple to fall + settle into stable contact with
# the plate (force > force_threshold AND velocity < velocity_threshold). Tuned empirically:
# a 0.02 m fall takes ~64 ms (~3 env steps @ 20 ms / step), then PhysX needs a handful of
# extra contact-resolution steps for the apple to come to rest. 50 steps = 1 s of sim time.
APPLE_SETTLE_STEPS = 50
HEADLESS = True
ENABLE_CAMERAS = True

APPLE_SCALE = (0.009, 0.009, 0.009)
PLATE_SCALE = (0.5, 0.5, 0.5)
APPLE_INITIAL_POSITION_M = (0.15, 0.15, 0.05)
PLATE_INITIAL_POSITION_M = (0.15, -0.40, 0.02)
# Drop height for the success-case teleport. Kept small so the apple has a short, contained
# fall onto the plate -- larger offsets risk the apple bouncing off a thin plate edge before
# settling into stable contact.
APPLE_ABOVE_PLATE_OFFSET_M = 0.02


def get_test_environment(num_envs: int):
    """Build a simplified G1 static apple-to-plate environment for testing.

    Uses a plain ``table`` background (instead of the production ``galileo_locomanip``
    scene) to isolate task-termination logic and keep the test fast. Mirrors the locomanip
    test's structure -- same standing-action pattern, same termination semantics -- so
    test failures here can be cleanly attributed to the static-task plumbing rather than
    the WBC stack itself.
    """

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    apple = asset_registry.get_asset_by_name("apple_01_objaverse_robolab")(scale=APPLE_SCALE)
    plate = asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")(scale=PLATE_SCALE)

    apple.set_initial_pose(Pose(position_xyz=APPLE_INITIAL_POSITION_M, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    plate.set_initial_pose(Pose(position_xyz=PLATE_INITIAL_POSITION_M, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    # Use the joint-control AGILE WBC variant in tests: the production env defaults to
    # ``g1_wbc_agile_pink`` (AGILE end-to-end velocity policy + PinkIK upper body), so
    # the joint twin matches the deployed WBC backend while skipping the PinkIK forward
    # solve on every step -- this test exercises task termination semantics, not the IK
    # stack. The zero-norm wrist quat bootstrapping fix in
    # ``g1_decoupled_wbc_pink_action._identity_if_zero_norm_xyzw`` makes the pink variant
    # safe to use here too, but the joint variant remains a deliberate speed optimization.
    embodiment = G1WBCAgileJointEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[background, apple, plate])
    # Construct ``PickAndPlaceTask`` with parent default termination thresholds: this test
    # exercises termination semantics, not threshold tuning. The production env keeps the
    # locomanip-tightened pair (0.5, 0.1) for comparable metrics; any drift in the parent
    # defaults is caught by the parent's own tests.
    task = PickAndPlaceTask(
        pick_up_object=apple,
        destination_location=plate,
        background_scene=background,
        episode_length_s=30.0,
        task_description="Pick up the apple from the table and place it onto the plate.",
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_g1_static_pick_and_place",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    return env, apple, plate


def _step_with_standing_actions(env, num_steps: int) -> list[bool]:
    """Step the env with standing-idle actions and return termination flags per step.

    Mirrors the locomanip test: zero actions everywhere except the hip-height channel
    (``actions[:, -4] = 0.75``), which tells WBC to hold standing height instead of
    interpreting the zero as "squat to floor". Identical action layout to the locomanip
    env because both pink variants and both joint variants share the same 23-D
    ``G1DecoupledWBCJointActionCfg`` -- only the lower-body ONNX backend (HOMIE vs AGILE)
    differs, and that does not change the action vector layout.
    """
    terminated_list = []
    for _ in range(num_steps):
        with torch.inference_mode():
            actions = _zero_actions(env)
            actions[:, -4] = 0.75
            _, _, terminated, _, _ = env.step(actions)
            terminated_list.append(terminated.item())
    return terminated_list


def _zero_actions(env) -> torch.Tensor:
    """Return a ``(num_envs, action_dim)``-shaped zero action tensor on the env's device.

    Uses ``single_action_space.shape`` instead of ``action_space.shape`` so the result
    has the correct rank regardless of how the wrapper exposes the vectorized action
    space (some wrappers prepend the batch dim, some don't).
    """
    return torch.zeros((env.unwrapped.num_envs,) + env.unwrapped.single_action_space.shape, device=env.unwrapped.device)


def _teleport_apple(env, apple, position_xyz: tuple[float, float, float]) -> None:
    """Teleport ``apple`` to ``position_xyz`` (env-local frame), identity rotation, zero velocity."""
    from isaaclab_arena.utils.pose import Pose

    with torch.inference_mode():
        apple.set_object_pose(env, Pose(position_xyz=position_xyz, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _test_initial_state_not_terminated(simulation_app) -> bool:
    """Apple starts away from the plate -- task must not be terminated."""

    env, apple, _ = get_test_environment(num_envs=1)

    try:
        # Warmup: let the WBC controller settle the standing pose before checking
        # termination. Same rationale as the locomanip test.
        _step_with_standing_actions(env, WARMUP_STEPS)
        _teleport_apple(env, apple, APPLE_INITIAL_POSITION_M)

        terminated_list = _step_with_standing_actions(env, NUM_STEPS)
        for step, terminated in enumerate(terminated_list):
            assert not terminated, f"Task terminated unexpectedly at post-warmup step {step}/{NUM_STEPS}"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def _test_apple_on_plate_succeeds(simulation_app) -> bool:
    """Teleporting the apple just above the plate once should trigger success termination as it settles.

    Single-teleport + settle pattern: the apple falls a small distance under gravity,
    rests on the plate, and the contact force + low velocity reliably trigger the
    termination within ``APPLE_SETTLE_STEPS``.
    """

    from isaaclab.assets import RigidObject

    env, apple, plate = get_test_environment(num_envs=1)

    try:
        _step_with_standing_actions(env, WARMUP_STEPS)

        plate_object: RigidObject = env.unwrapped.scene[plate.name]
        plate_pos_world = wp.to_torch(plate_object.data.root_pos_w)[0]
        env_origin = env.unwrapped.scene.env_origins[0]
        # ``wp.to_torch`` may return a tensor on a different device than ``env_origins``
        # (warp-managed vs torch-managed); explicitly align before subtracting.
        plate_pos_local = plate_pos_world.to(env_origin.device) - env_origin
        apple_target = (
            float(plate_pos_local[0]),
            float(plate_pos_local[1]),
            float(plate_pos_local[2]) + APPLE_ABOVE_PLATE_OFFSET_M,
        )

        # One-shot teleport: drop the apple above the plate, then run physics until it
        # settles into stable contact (or we hit APPLE_SETTLE_STEPS).
        _teleport_apple(env, apple, apple_target)

        terminated_list = _step_with_standing_actions(env, APPLE_SETTLE_STEPS)
        terminated_ever = any(terminated_list)

        assert terminated_ever, (
            "Task should terminate after apple is placed on plate; got terminated_list="
            f"{terminated_list[:10]}... (showing first 10 of {len(terminated_list)})"
        )
        print(f"Success: apple-on-plate termination detected (fired at step {terminated_list.index(True)})")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


@pytest.mark.with_cameras
def test_initial_state_not_terminated():
    result = run_simulation_app_function(
        _test_initial_state_not_terminated,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_initial_state_not_terminated.__name__} failed"


@pytest.mark.with_cameras
def test_apple_on_plate_succeeds():
    result = run_simulation_app_function(
        _test_apple_on_plate_succeeds,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_apple_on_plate_succeeds.__name__} failed"


if __name__ == "__main__":
    test_initial_state_not_terminated()
    test_apple_on_plate_succeeds()
