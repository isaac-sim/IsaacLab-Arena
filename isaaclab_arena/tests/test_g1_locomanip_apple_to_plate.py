# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import traceback

import pytest
import warp as wp

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
WARMUP_STEPS = 50
HEADLESS = True
ENABLE_CAMERAS = True

APPLE_INITIAL_POSITION_M = (0.15, 0.15, 0.05)
PLATE_INITIAL_POSITION_M = (0.15, -0.40, 0.02)
APPLE_ABOVE_PLATE_OFFSET_M = 0.05

# Match the tighter apple-on-plate proximity guard used by the production locomanip env so the test
# exercises the same ``success_proximity_max_distance`` value that actually ships. See
# ``_SUCCESS_PROXIMITY_OVERRIDES_M`` in ``galileo_g1_locomanip_pick_and_place_environment``.
APPLE_ON_PLATE_PROXIMITY_M = 0.10


def get_test_environment(num_envs: int):
    """Build a simplified G1 locomanip apple-to-plate environment for testing.

    Uses a plain table background (instead of the full ``galileo_locomanip`` scene) to isolate
    task termination logic from the production environment while still exercising the
    ``LocomanipPickAndPlaceTask`` + G1 WBC locomotion embodiment stack used by
    ``galileo_g1_locomanip_pick_and_place``.
    """

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.g1.g1 import G1WBCJointEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    apple = asset_registry.get_asset_by_name("apple_01_objaverse_robolab")()
    plate = asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")()

    apple.set_initial_pose(Pose(position_xyz=APPLE_INITIAL_POSITION_M, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    plate.set_initial_pose(Pose(position_xyz=PLATE_INITIAL_POSITION_M, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    embodiment = G1WBCJointEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[background, apple, plate])
    task = LocomanipPickAndPlaceTask(
        pick_up_object=apple,
        destination_location=plate,
        background_scene=background,
        episode_length_s=30.0,
        task_description="Pick up the apple from the table and place it onto the plate.",
        success_proximity_max_distance=APPLE_ON_PLATE_PROXIMITY_M,
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_g1_locomanip_apple_to_plate",
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
    """Step the environment with standing idle actions and return termination flags."""
    terminated_list = []
    for _ in range(num_steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # NOTE: Set base height to 0.75m to avoid robot squatting to match 0-height command.
            actions[:, -4] = 0.75
            _, _, terminated, _, _ = env.step(actions)
            terminated_list.append(terminated.item())
    return terminated_list


def _teleport_apple(env, apple, position_xyz: tuple[float, float, float]) -> None:
    """Teleport ``apple`` to ``position_xyz`` (env-local frame) with identity orientation and zero velocity.

    Uses ``ObjectBase.set_object_pose`` so we piggyback on the framework helper for pose + velocity
    resets instead of reaching into the raw ``RigidObject.write_root_*`` APIs. Note the ``Pose`` quat
    convention is ``xyzw``, so ``(0, 0, 0, 1)`` is the identity quaternion (Isaac Lab 3.0).
    """
    from isaaclab_arena.utils.pose import Pose

    with torch.inference_mode():
        apple.set_object_pose(env, Pose(position_xyz=position_xyz, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _test_initial_state_not_terminated(simulation_app) -> bool:
    """Apple starts away from the plate -- task must not be terminated."""

    env, apple, _ = get_test_environment(num_envs=1)

    try:
        # Warmup: let the G1 WBC policy stabilise the robot before checking termination. During the
        # first few dozen sim steps the lower-body controller settles, which can cause brief physics
        # transients (vibrations, contacts) that may nudge the apple and trigger spurious termination.
        _step_with_standing_actions(env, WARMUP_STEPS)

        # Re-place the apple after warmup; the robot's stabilisation can knock it off the table and
        # trigger the object_dropped termination before we even start the real assertion steps.
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
    """Re-teleporting the apple above the plate should trigger success termination."""

    from isaaclab.assets import RigidObject

    env, apple, plate = get_test_environment(num_envs=1)

    try:
        _step_with_standing_actions(env, WARMUP_STEPS)

        plate_object: RigidObject = env.unwrapped.scene[plate.name]
        plate_pos_world = wp.to_torch(plate_object.data.root_pos_w)[0]
        env_origin = env.unwrapped.scene.env_origins[0]
        plate_pos_local = plate_pos_world - env_origin
        apple_target = (
            float(plate_pos_local[0]),
            float(plate_pos_local[1]),
            float(plate_pos_local[2]) + APPLE_ABOVE_PLATE_OFFSET_M,
        )

        terminated_ever = False
        for _ in range(NUM_STEPS * 10):
            # Re-teleport each step so short physics drifts (bouncing off a thin plate edge)
            # cannot push the apple outside the proximity threshold before termination fires.
            _teleport_apple(env, apple, apple_target)
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                actions[:, -4] = 0.75
                _, _, terminated, _, _ = env.step(actions)
            if terminated.item():
                terminated_ever = True
                break

        assert terminated_ever, "Task should terminate after apple is placed on plate"
        print("Success: apple-on-plate termination detected")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_mimic_cfg_uses_object_and_destination_names(simulation_app) -> bool:
    """Verify the Mimic config picks up both the object and destination names from the task.

    Guards against silent regressions where the Mimic ``SubTaskConfig.object_ref`` or the
    ``datagen_config.name`` stay wired to the old hardcoded ``brown_box`` / ``blue_sorting_bin``
    identifiers instead of following the constructor arguments.
    """

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    apple = asset_registry.get_asset_by_name("apple_01_objaverse_robolab")()
    plate = asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")()
    background = asset_registry.get_asset_by_name("table")()

    apple.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    plate.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

    task = LocomanipPickAndPlaceTask(
        pick_up_object=apple,
        destination_location=plate,
        background_scene=background,
    )

    mimic_cfg = task.get_mimic_env_cfg(arm_mode=ArmMode.DUAL_ARM)

    assert (
        mimic_cfg.pick_up_object_name == apple.name
    ), f"Expected pick_up_object_name='{apple.name}', got '{mimic_cfg.pick_up_object_name}'"

    assert (
        mimic_cfg.destination_name == plate.name
    ), f"Expected destination_name='{plate.name}', got '{mimic_cfg.destination_name}'"

    # Datagen name must include BOTH the object and the destination so Mimic runs for e.g.
    # apple+plate vs apple+bin don't collide on the same dataset key.
    assert (
        apple.name in mimic_cfg.datagen_config.name
    ), f"Expected datagen_config.name to include '{apple.name}', got '{mimic_cfg.datagen_config.name}'"
    assert (
        plate.name in mimic_cfg.datagen_config.name
    ), f"Expected datagen_config.name to include '{plate.name}', got '{mimic_cfg.datagen_config.name}'"

    for arm_key in ("right", "left", "body"):
        for i, subtask in enumerate(mimic_cfg.subtask_configs[arm_key]):
            assert (
                subtask.object_ref == apple.name
            ), f"subtask_configs['{arm_key}'][{i}].object_ref should be '{apple.name}', got '{subtask.object_ref}'"

    print("Success: Mimic config correctly uses apple object + plate destination names")
    return True


def _test_mimic_cfg_brown_box_preserves_legacy_datagen_name(simulation_app) -> bool:
    """Regression guard: the default brown_box+blue_sorting_bin Mimic cfg must keep the main datagen name.

    Existing Mimic datasets and policy checkpoints are keyed on ``"locomanip_pick_and_place_D0"``
    (post-#571 refactor); any drift here silently invalidates them. The legacy name is reserved
    for the exact (brown_box, blue_sorting_bin) pair only -- any other pairing must use the
    per-pair templated name (covered by the apple test above and the
    brown_box+non_default_destination test below).
    """

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    brown_box = asset_registry.get_asset_by_name("brown_box")()
    blue_bin = asset_registry.get_asset_by_name("blue_sorting_bin")()
    background = asset_registry.get_asset_by_name("table")()

    brown_box.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    blue_bin.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

    task = LocomanipPickAndPlaceTask(
        pick_up_object=brown_box,
        destination_location=blue_bin,
        background_scene=background,
    )

    mimic_cfg = task.get_mimic_env_cfg(arm_mode=ArmMode.DUAL_ARM)

    expected_name = "locomanip_pick_and_place_D0"
    assert mimic_cfg.datagen_config.name == expected_name, (
        f"brown_box+blue_sorting_bin Mimic datagen name must stay '{expected_name}' for backward "
        f"compatibility, got '{mimic_cfg.datagen_config.name}'"
    )

    print(f"Success: brown_box+blue_sorting_bin Mimic cfg preserves legacy datagen name '{expected_name}'")
    return True


def _test_mimic_cfg_brown_box_non_default_destination_is_not_legacy(simulation_app) -> bool:
    """Only the exact legacy (brown_box, blue_sorting_bin) pair should get the legacy datagen name.

    ``brown_box`` paired with a non-default destination (e.g. the clay plate) must generate its
    own per-pair datagen key so Mimic runs don't silently overwrite the brown_box dataset.
    """

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    brown_box = asset_registry.get_asset_by_name("brown_box")()
    plate = asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")()
    background = asset_registry.get_asset_by_name("table")()

    brown_box.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    plate.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

    task = LocomanipPickAndPlaceTask(
        pick_up_object=brown_box,
        destination_location=plate,
        background_scene=background,
    )

    mimic_cfg = task.get_mimic_env_cfg(arm_mode=ArmMode.DUAL_ARM)

    legacy_name = "locomanip_pick_and_place_D0"
    assert mimic_cfg.datagen_config.name != legacy_name, (
        f"brown_box with a non-default destination ('{plate.name}') must NOT reuse the legacy "
        f"'{legacy_name}' datagen name, but got exactly that."
    )
    assert brown_box.name in mimic_cfg.datagen_config.name and plate.name in mimic_cfg.datagen_config.name, (
        f"Expected datagen_config.name to include both '{brown_box.name}' and '{plate.name}', "
        f"got '{mimic_cfg.datagen_config.name}'"
    )

    print(f"Success: brown_box+non-default destination uses per-pair datagen name '{mimic_cfg.datagen_config.name}'")
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


def test_mimic_cfg_uses_object_and_destination_names():
    result = run_simulation_app_function(
        _test_mimic_cfg_uses_object_and_destination_names,
        headless=HEADLESS,
        enable_cameras=False,
    )
    assert result, f"Test {_test_mimic_cfg_uses_object_and_destination_names.__name__} failed"


def test_mimic_cfg_brown_box_preserves_legacy_datagen_name():
    result = run_simulation_app_function(
        _test_mimic_cfg_brown_box_preserves_legacy_datagen_name,
        headless=HEADLESS,
        enable_cameras=False,
    )
    assert result, f"Test {_test_mimic_cfg_brown_box_preserves_legacy_datagen_name.__name__} failed"


def test_mimic_cfg_brown_box_non_default_destination_is_not_legacy():
    result = run_simulation_app_function(
        _test_mimic_cfg_brown_box_non_default_destination_is_not_legacy,
        headless=HEADLESS,
        enable_cameras=False,
    )
    assert result, f"Test {_test_mimic_cfg_brown_box_non_default_destination_is_not_legacy.__name__} failed"


if __name__ == "__main__":
    test_initial_state_not_terminated()
    test_apple_on_plate_succeeds()
    test_mimic_cfg_uses_object_and_destination_names()
    test_mimic_cfg_brown_box_preserves_legacy_datagen_name()
    test_mimic_cfg_brown_box_non_default_destination_is_not_legacy()
