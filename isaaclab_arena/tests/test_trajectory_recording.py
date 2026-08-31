# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check that the trajectory recorder terms reach the exported dataset."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

# Episodes only end on success or on the task's own timeout, which a zero action never reaches, so
# the rollout is short and the episode boundary is forced by resetting explicitly.
NUM_STEPS = 30
NUM_ENVS = 2
HEADLESS = True

DATASET_FILENAME = "dataset_trajectory_recording"

# Distance the end effector is allowed to sit from its own robot base. Poses are recorded
# env-relative, so an env's offset in the world must not show up in them.
MAX_END_EFFECTOR_DISTANCE_FROM_BASE_M = 2.0

# Recorded EE linear velocity vs finite difference of position. Instantaneous sim velocity and a
# discrete derivative disagree by O(dt); peak errors around 0.2 m/s are expected under IK motion.
MAX_LINEAR_VELOCITY_FINITE_DIFF_ERROR_M_S = 0.25
MAX_LINEAR_VELOCITY_FINITE_DIFF_MEDIAN_ERROR_M_S = 0.12
# Reject a motionless rollout: otherwise zeros match zeros and the check is vacuous.
MIN_EE_LINEAR_SPEED_FOR_VELOCITY_CHECK_M_S = 0.05


def _create_trajectory_recording_env(output_dir):
    """Build a two-env pick-and-place env whose recorders include the trajectory terms.

    Args:
        output_dir: Directory the exported dataset is written into.
    """
    from dataclasses import replace

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )

    scene = Scene(assets=[background, cracker_box])
    arena_environment = IsaacLabArenaEnvironment(
        name="trajectory_recording",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    builder_cfg = replace(
        arena_env_builder_cfg_from_argparse(args_cli),
        record_trajectories=True,
        recorder_dataset_export_dir_path=str(output_dir),
        recorder_dataset_filename=DATASET_FILENAME,
    )
    env_builder = ArenaEnvBuilder(arena_environment, builder_cfg)
    env_cfg, env_kwargs = env_builder.compose_manager_cfg()

    env = env_builder.make_registered(env_cfg, env_kwargs)
    env.unwrapped.episode_recorder.set_job_name("trajectory_recording")
    env.unwrapped.episode_recorder.set_output_path(Path(output_dir) / "episode_results.jsonl")
    env.reset()
    return env


def _roll_out_and_end_the_episode(env):
    """Step the env with non-zero actions, then reset so the recorder exports the episode.

    Non-zero commands are required so the EE actually moves; otherwise a finite-difference
    check of linear velocity would pass vacuously on near-zero signals.
    """
    import torch
    import tqdm

    device = env.unwrapped.device
    # Deterministic open-loop command: drive the first translational action channel.
    action = torch.zeros(env.action_space.shape, device=device)
    action[:, 0] = 0.2

    with torch.inference_mode():
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            env.step(action)
        env.reset()


def _assert_linear_velocity_matches_finite_difference(end_effector, step_dt: float) -> None:
    """Check recorded EE linear velocity against finite differences of EE position.

    Instantaneous sim velocity and a discrete derivative of position disagree by O(dt), and
    the best matching one-sided vs central difference depends on when velocity is sampled
    within the step. The check therefore takes the best of forward, backward, and central
    differences at each interior sample.

    Args:
        end_effector: HDF5 group with position and linear_velocity datasets.
        step_dt: Environment step duration used when the demo was recorded [s].
    """
    import numpy as np

    position = end_effector["position"][:]
    linear_velocity = end_effector["linear_velocity"][:]
    assert position.shape[0] >= 3, "Need at least 3 steps for a finite-difference check"

    velocity_rec = linear_velocity[1:-1]
    velocity_fd_central = (position[2:] - position[:-2]) / (2.0 * step_dt)
    velocity_fd_forward = (position[2:] - position[1:-1]) / step_dt
    velocity_fd_backward = (position[1:-1] - position[:-2]) / step_dt
    error = np.minimum(
        np.linalg.norm(velocity_rec - velocity_fd_central, axis=-1),
        np.minimum(
            np.linalg.norm(velocity_rec - velocity_fd_forward, axis=-1),
            np.linalg.norm(velocity_rec - velocity_fd_backward, axis=-1),
        ),
    )
    peak_recorded_speed = np.linalg.norm(linear_velocity, axis=-1).max()
    assert (
        peak_recorded_speed > MIN_EE_LINEAR_SPEED_FOR_VELOCITY_CHECK_M_S
    ), f"EE barely moved (peak speed {peak_recorded_speed:.4f} m/s); finite-difference check is vacuous"
    assert error.max() < MAX_LINEAR_VELOCITY_FINITE_DIFF_ERROR_M_S, (
        "EE linear_velocity disagrees with finite-difference of position: "
        f"max |v_rec - v_fd|={error.max():.4f} m/s (limit {MAX_LINEAR_VELOCITY_FINITE_DIFF_ERROR_M_S})"
    )
    assert float(np.median(error)) < MAX_LINEAR_VELOCITY_FINITE_DIFF_MEDIAN_ERROR_M_S, (
        "EE linear_velocity median finite-difference error too high: "
        f"{float(np.median(error)):.4f} m/s (limit {MAX_LINEAR_VELOCITY_FINITE_DIFF_MEDIAN_ERROR_M_S})"
    )


def _test_trajectory_terms_reach_the_dataset(simulation_app, output_dir):  # noqa: ARG001
    import h5py
    import numpy as np

    env = _create_trajectory_recording_env(output_dir)
    step_dt = float(env.unwrapped.step_dt)
    try:
        _roll_out_and_end_the_episode(env)
    finally:
        # Closing flushes any episode still open, so read the dataset afterwards.
        env.close()

    dataset_path = Path(output_dir) / f"{DATASET_FILENAME}.hdf5"
    assert dataset_path.exists(), f"Expected an exported dataset at {dataset_path}"

    with h5py.File(dataset_path, "r") as dataset:
        demos = [dataset["data"][name] for name in dataset["data"]]
        assert demos, "Expected at least one exported demo"

        for demo in demos:
            num_steps = demo["actions"].shape[0]

            # Each demo carries the (env_id, episode_in_env) pair that identifies it in the JSONL.
            assert demo["episode_id/env_id"].shape == (1,)
            assert demo["episode_id/episode_in_env"].shape == (1,)

            end_effector = demo["states/kinematics/end_effector"]
            assert end_effector["position"].shape == (num_steps, 3)
            assert end_effector["orientation"].shape == (num_steps, 4)
            assert end_effector["linear_velocity"].shape == (num_steps, 3)
            assert end_effector["angular_velocity"].shape == (num_steps, 3)
            assert np.isfinite(end_effector["linear_velocity"][:]).all()
            assert np.isfinite(end_effector["angular_velocity"][:]).all()
            _assert_linear_velocity_matches_finite_difference(end_effector, step_dt)

            # Initial-state kinematics include poses and velocities, but not gripper commands.
            initial_ee = demo["initial_state/kinematics/end_effector"]
            assert initial_ee["linear_velocity"].shape == (1, 3)
            assert initial_ee["angular_velocity"].shape == (1, 3)

            # The gripper's commanded state is only known once an action has been processed, so it
            # is recorded per step but not for the initial state.
            kinematics = demo["states/kinematics"]
            gripper_names = [name for name in kinematics if "is_commanded_open" in kinematics[name]]
            assert gripper_names, f"Expected a gripper group with is_commanded_open, got {list(kinematics)}"
            for gripper_name in gripper_names:
                # One column per driver joint, so the width is embodiment-specific.
                opening = kinematics[gripper_name]["position"]
                assert opening.shape[0] == num_steps
                assert np.isfinite(opening[:]).all()
                assert kinematics[gripper_name]["is_commanded_open"].dtype == bool
                assert "is_commanded_open" not in demo[f"initial_state/kinematics/{gripper_name}"]

            # The task's metric terms must survive alongside the trajectory terms, because metrics
            # are computed by reading them back out of this same dataset.
            assert "success" in demo

            # Poses are env-relative, so the end effector stays near its own robot base however far
            # the env is offset in the world.
            distance_from_base = np.abs(
                end_effector["position"][:] - demo["states/articulation/robot/root_pose"][:, :3]
            ).max()
            assert (
                distance_from_base < MAX_END_EFFECTOR_DISTANCE_FROM_BASE_M
            ), f"End-effector poses look world-absolute: {distance_from_base:.2f} m from the robot base"

    return True


def test_trajectory_terms_reach_the_dataset(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_trajectory_terms_reach_the_dataset,
        headless=HEADLESS,
        output_dir=tmp_path,
    )
    assert result, f"Test {_test_trajectory_terms_reach_the_dataset.__name__} failed"
