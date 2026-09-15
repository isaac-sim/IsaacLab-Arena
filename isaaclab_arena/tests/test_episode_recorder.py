# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import torch
import tqdm
from dataclasses import field
from pathlib import Path

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

NUM_STEPS = 200
NUM_ENVS = 2
HEADLESS = True

JOB_NAME = "unit_test"
LANGUAGE_INSTRUCTION = "put the box in the drawer"

# Fields stamped by the manager (metadata) plus those from the default core term.
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

# Field contributed by the custom term registered in the custom-term test.
CUSTOM_KEY = "step_bucket"

# Field contributed by the progress-tracking recorder.
PROGRESS_KEY = "progress"

# Deterministic, single-valued (low == high) sample for the variation test, so each draw is known.
VARIATION_NAME = "record_test_variation"
VARIATION_SAMPLE = [0.25, 0.5]


def record_step_bucket(env, env_id):
    """Custom recorder term: records the finished episode's length bucketed into tens."""
    return {CUSTOM_KEY: int(env.episode_length_buf[env_id].item()) // 10}


def draw_record_test_variation(env, env_ids, asset_cfg, sampler):  # noqa: ARG001
    """Reset event that only draws a sample, so the variation recorder attributes it to the episode."""
    sampler.sample(num_samples=len(env_ids), env_ids=env_ids)


@configclass
class RecordTestVariationCfg(VariationBaseCfg):
    """Cfg for ``RecordTestVariation`` with a degenerate (constant) sampler for deterministic draws."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=VARIATION_SAMPLE, high=VARIATION_SAMPLE),
    )


class RecordTestVariation(RunTimeVariationBase):
    """Minimal run-time variation that samples on each reset without mutating the scene."""

    cfg: RecordTestVariationCfg

    def __init__(self, asset_name: str, name: str = VARIATION_NAME):
        super().__init__(cfg=RecordTestVariationCfg(), name=name)
        self.asset_name = asset_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        event_cfg = EventTermCfg(
            func=draw_record_test_variation,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(self.asset_name), "sampler": self._sampler},
        )
        return f"{self.asset_name}_{VARIATION_NAME}", event_cfg


def create_recorder_env(
    output_dir, *, episode_recorder_terms: dict[str, object] | None = None, enable_variation: bool = False
):
    """Build a registered two-env pick-and-place env wired for per-episode recording.

    The rollout lifts and replaces env 0's box in the drawer. Env 1's box falls outside it.

    Args:
        output_dir: Directory the JSONL records are written into.
        episode_recorder_terms: Extra per-episode recorder terms (i.e. EpisodeRecorderTermCfg.
        enable_variation: When True, attach an enabled run-time variation to the cracker box.

    Returns:
        An ``(env, output_path)`` tuple: the registered env and the JSONL path to write the records to.
    """
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.terms.events import set_object_pose_per_env
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )

    if enable_variation:
        variation = RecordTestVariation(cracker_box.name)
        variation.enable()
        cracker_box.add_variation(variation)

    scene = Scene(assets=[background, cracker_box, destination_location])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="episode_recorder",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
        episode_recorder_terms=episode_recorder_terms or {},
    )

    # The builder applies the language-instruction override onto the env cfg's task_description, which the
    # core recorder then records.
    env_builder = ArenaEnvBuilder(
        isaaclab_arena_environment,
        ArenaEnvBuilderCfg(num_envs=NUM_ENVS, language_instruction=LANGUAGE_INSTRUCTION),
    )
    env_cfg, env_kwargs = env_builder.compose_manager_cfg()

    # Env 0 starts in the drawer and is lifted during rollout; env 1 lands outside.
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

    output_path = Path(output_dir) / "episode_results.jsonl"

    env = env_builder.make_registered(env_cfg, env_kwargs)
    env.unwrapped.episode_recorder.set_job_name(JOB_NAME)
    env.unwrapped.episode_recorder.set_output_path(output_path)
    env.reset()
    return env, output_path


def _roll_out_and_read_episode_record(env, output_path) -> list[dict]:
    """Lift env 0's settled box each episode, let it land, and read the episode records."""
    from isaaclab_arena.tests.utils.pick_and_place import lift_settled_objects_once

    base_env = env.unwrapped
    object_name = base_env.cfg.events.reset_pick_up_object_pose.params["asset_cfg"].name
    # Env 1 never receives a lift; its episodes should fail.
    lifted_envs = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    previous_episode = None
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            current_episode = base_env.get_episode_index(0)
            if current_episode != previous_episode:
                lifted_envs[0] = False
                previous_episode = current_episode
            lift_settled_objects_once(base_env, object_name, lifted_envs)
            actions = torch.zeros(env.action_space.shape, device=base_env.device)
            env.step(actions)

    assert output_path.exists(), f"Expected JSONL at {output_path}"
    with open(output_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Recorded {len(records)} episode(s)")
    return records


def _test_core_terms(simulation_app, output_dir):  # noqa: ARG001
    env, output_path = create_recorder_env(output_dir)
    try:
        records = _roll_out_and_read_episode_record(env, output_path)
        assert len(records) >= NUM_ENVS, f"Expected at least {NUM_ENVS} episodes, got {len(records)}"

        # episode_in_env must increment from 0 per env, and the deterministic poses fix success.
        per_env_counter: dict[int, int] = {}
        for record in records:
            # With no variation drawn and no custom term, every record is the core schema plus the
            # progress block contributed by PickAndPlaceTask's progress objectives.
            expected_keys = CORE_KEYS | {PROGRESS_KEY}
            assert set(record.keys()) == expected_keys, f"Unexpected keys: {set(record.keys()) ^ expected_keys}"
            assert record["job_name"] == JOB_NAME
            assert record["language_instruction"] == LANGUAGE_INSTRUCTION
            assert isinstance(record["episode_length"], int)

            env_id = record["env_id"]
            assert env_id in (0, 1)
            assert record["episode_in_env"] == per_env_counter.get(env_id, 0)
            per_env_counter[env_id] = per_env_counter.get(env_id, 0) + 1
            expected_success = env_id == 0
            assert (
                record["success"] is expected_success
            ), f"env {env_id} episode {record['episode_in_env']}: expected success={expected_success}"
            assert record["progress"]["all_complete"] is expected_success
            if expected_success:
                assert [event["predicate_index"] for event in record["progress"]["events"]] == [0, 1, 2]

        # Both envs must have completed at least one episode.
        assert set(per_env_counter.keys()) == {0, 1}
    finally:
        env.close()
    return True


def _test_variations_recorded(simulation_app, output_dir):  # noqa: ARG001
    env, output_path = create_recorder_env(output_dir, enable_variation=True)
    try:
        records = _roll_out_and_read_episode_record(env, output_path)

        # The enabled variation must be registered with the recorder and recorded on every episode.
        recorded_keys = set(env.unwrapped.variation_recorder.records.keys())
        assert recorded_keys, "Expected the enabled variation to be attached to the variation recorder"
        for record in records:
            assert "variations" in record, f"Missing 'variations' field: {set(record.keys())}"
            assert set(record["variations"].keys()) == recorded_keys
            for value in record["variations"].values():
                assert value == VARIATION_SAMPLE, f"Expected sample {VARIATION_SAMPLE}, got {value}"
    finally:
        env.close()
    return True


def _test_custom_term(simulation_app, output_dir):  # noqa: ARG001
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    custom_terms = {"step_bucket": EpisodeRecorderTermCfg(func=record_step_bucket)}
    env, output_path = create_recorder_env(output_dir, episode_recorder_terms=custom_terms)
    try:
        records = _roll_out_and_read_episode_record(env, output_path)

        # The custom term's field is present and derived from the same intact episode-length buffer.
        for record in records:
            expected_keys = CORE_KEYS | {PROGRESS_KEY, CUSTOM_KEY}
            assert set(record.keys()) == expected_keys, f"Unexpected keys: {set(record.keys()) ^ expected_keys}"
            assert record[CUSTOM_KEY] == record["episode_length"] // 10
    finally:
        env.close()
    return True


def test_core_terms(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_core_terms, headless=HEADLESS, output_dir=tmp_path
    ), "core recorder terms test failed"


def test_variations_recorded(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_variations_recorded, headless=HEADLESS, output_dir=tmp_path
    ), "variation recording test failed"


def test_custom_term(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_custom_term, headless=HEADLESS, output_dir=tmp_path
    ), "custom recorder term test failed"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="episode_recorder_") as _tmp_dir:
        test_core_terms(Path(_tmp_dir))
        test_variations_recorded(Path(_tmp_dir))
        test_custom_term(Path(_tmp_dir))
