# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the ``EpisodeRecorderManager`` per-episode recording.

Two envs roll deterministic episodes (env 0 always succeeds, env 1 always fails). A custom recorder
term is declared through the cfg; we set the run-level metadata after the fact, roll, then request a
write (passing in the path) and assert the JSONL file has one well-formed line per completed
episode, with the success flag matching each env and the custom term's field present -- exercising
the ``EpisodeRecorderTermCfg`` extension point.
"""

import json
import tempfile
import torch
import tqdm
import traceback
from pathlib import Path

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 200
HEADLESS = True

# Fields stamped by the manager (metadata + indices) plus those from the default core term.
CORE_KEYS = {
    "job_name",
    "episode_in_env",
    "env_id",
    "seed",
    "success",
    "episode_length",
    "language_instruction",
    "timestamp",
}
# Field contributed by the custom term registered in this test.
CUSTOM_KEY = "step_bucket"


def _test_episode_recorder(simulation_app):
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.evaluation.episode_recorder import EpisodeRecorderTermCfg, EpisodeResultsMetadata
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.terms.events import set_object_pose_per_env
    from isaaclab_arena.utils.pose import Pose

    def record_step_bucket(env, context):
        """Custom term callable: records the finished episode's length bucketed into tens."""
        return {CUSTOM_KEY: int(env.episode_length_buf[context.env_id].item()) // 10}

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
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="episode_recorder",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )

    NUM_ENVS = 2
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env_cfg, env_kwargs = env_builder.compose_manager_cfg()

    # Per-env reset poses: env 0 lands in the drawer (success), env 1 lands outside (failure).
    pose_list = [
        Pose(position_xyz=(0.0, -0.5, 0.2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        Pose(position_xyz=(-0.5, -0.5, 0.2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    ]
    env_cfg.events.reset_pick_up_object_pose = EventTermCfg(
        func=set_object_pose_per_env,
        mode="reset",
        params={
            "pose_list": pose_list,
            "asset_cfg": SceneEntityCfg(cracker_box.name),
        },
    )

    # Register a custom episode recorder term through the cfg, exercising the extension point.
    env_cfg.episode_recorders.step_bucket = EpisodeRecorderTermCfg(func=record_step_bucket)

    tmp_dir = tempfile.mkdtemp(prefix="episode_recorder_")
    output_path = Path(tmp_dir) / "episode_results.jsonl"

    env = env_builder.make_registered(env_cfg, env_kwargs)

    # Set the run-level metadata on the dedicated episode recorder manager after the fact.
    env.unwrapped.episode_recorder.set_metadata(
        EpisodeResultsMetadata(
            job_name="unit_test",
            language_instruction="put the box in the drawer",
        )
    )
    env.reset()

    try:
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Request the write from outside the env, passing in the output path.
        env.unwrapped.episode_recorder.write(output_path)

        assert output_path.exists(), f"Expected JSONL at {output_path}"
        with open(output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        print(f"Recorded {len(records)} episode(s)")
        assert len(records) >= NUM_ENVS, f"Expected at least {NUM_ENVS} episodes, got {len(records)}"

        expected_keys = CORE_KEYS | {CUSTOM_KEY}
        # Schema (core + custom term) and the constant run-level fields.
        for record in records:
            assert expected_keys == set(record.keys()), f"Unexpected keys: {set(record.keys())}"
            assert record["job_name"] == "unit_test"
            assert record["language_instruction"] == "put the box in the drawer"
            assert record["env_id"] in (0, 1)
            assert isinstance(record["episode_length"], int)
            # The custom term's field is derived from the same intact episode-length buffer.
            assert record[CUSTOM_KEY] == record["episode_length"] // 10

        # episode_in_env must increment from 0 per env, and the deterministic poses fix success.
        per_env_counter: dict[int, int] = {}
        for record in records:
            env_id = record["env_id"]
            assert record["episode_in_env"] == per_env_counter.get(env_id, 0)
            per_env_counter[env_id] = per_env_counter.get(env_id, 0) + 1
            expected_success = env_id == 0
            assert (
                record["success"] is expected_success
            ), f"env {env_id} episode {record['episode_in_env']}: expected success={expected_success}"

        # Both envs must have completed at least one episode.
        assert set(per_env_counter.keys()) == {0, 1}

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()
        for path in Path(tmp_dir).glob("*"):
            path.unlink(missing_ok=True)
        Path(tmp_dir).rmdir()

    return True


def test_episode_recorder():
    result = run_simulation_app_function(
        _test_episode_recorder,
        headless=HEADLESS,
    )
    assert result, f"Test {test_episode_recorder.__name__} failed"


if __name__ == "__main__":
    test_episode_recorder()
