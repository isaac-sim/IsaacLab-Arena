# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import torch
import tqdm
from contextlib import nullcontext
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
    output_dir,
    *,
    episode_recorder_terms: dict[str, object] | None = None,
    enable_variation: bool = False,
    task_type=None,
    reset_on_create: bool = True,
):
    """Build a registered two-env pick-and-place env wired for per-episode recording.

    env 0's box lands in the drawer (success) while env 1's box lands outside it (failure).

    Args:
        output_dir: Directory the JSONL records are written into.
        episode_recorder_terms: Extra per-episode recorder terms (EpisodeRecorderTermCfg).
        enable_variation: When True, attach an enabled run-time variation to the cracker box.
        task_type: Optional PickAndPlaceTask subclass that contributes recorder terms.
        reset_on_create: Whether to perform the initial reset before returning the environment.

    Returns:
        An ``(env, output_path)`` tuple: the registered env and the JSONL path to write the records to.
    """
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
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
        task=(task_type or PickAndPlaceTask)(cracker_box, destination_location, background),
        teleop_device=None,
        episode_recorder_terms=episode_recorder_terms or {},
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    # The builder applies the language-instruction override onto the env cfg's task_description, which the
    # core recorder then records.
    args_cli.language_instruction = LANGUAGE_INSTRUCTION
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
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

    output_path = Path(output_dir) / "episode_results.jsonl"

    env = env_builder.make_registered(env_cfg, env_kwargs)
    env.unwrapped.episode_recorder.set_job_name(JOB_NAME)
    env.unwrapped.episode_recorder.set_output_path(output_path)
    try:
        if reset_on_create:
            env.reset()
    except Exception:
        env.close()
        raise
    return env, output_path


def _roll_out_and_read_episode_record(env, output_path) -> list[dict]:
    """Step the env for ``NUM_STEPS`` (records stream to disk as episodes finish), then parse them."""
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
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


def _test_task_recorder_lifecycle(simulation_app, output_dir, initial_reset_mode):  # noqa: ARG001
    from isaaclab.managers import ManagerTermBase

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    notifications = []
    reset_lengths = []
    recorder_terms = []

    class StartingPoseRecorder(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.starting_poses = {}
            self.starting_indices = {}
            recorder_terms.append(self)

        def reset(self, env_ids=None):
            ids = list(range(self.num_envs)) if env_ids is None else [int(i) for i in env_ids]
            notifications.append(ids)
            asset = self._env.scene[self.cfg.params["asset_name"]]
            for env_id in ids:
                episode_index = self._env.get_episode_index(env_id)
                episode_length = int(self._env.episode_length_buf[env_id])
                reset_lengths.append((env_id, episode_index, episode_length))
                self.starting_poses[env_id] = asset.data.root_pose_w.torch[env_id].tolist()
                self.starting_indices[env_id] = episode_index

        def __call__(self, env, env_id, asset_name):  # noqa: ARG002
            return {
                "starting_pose": self.starting_poses[env_id],
                "starting_episode_index": self.starting_indices[env_id],
            }

    class RecordingTask(PickAndPlaceTask):
        def get_metrics(self):
            # SuccessRateMetric separately requires a full first reset; isolate the recorder contract here.
            return [] if initial_reset_mode == "partial_reset_to" else super().get_metrics()

        def get_episode_recorder_terms(self, arena_env):
            assert arena_env.task is self
            return {
                "starting_pose": EpisodeRecorderTermCfg(
                    func=StartingPoseRecorder, params={"asset_name": self.pick_up_object.name}
                )
            }

    env, output_path = create_recorder_env(
        output_dir,
        task_type=RecordingTask,
        episode_recorder_terms={"step_bucket": EpisodeRecorderTermCfg(func=record_step_bucket)},
        enable_variation=True,
        reset_on_create=False,
    )
    try:
        base_env = env.unwrapped
        assert len(recorder_terms) == 1, "The task recorder must be constructed exactly once"
        term = recorder_terms[0]
        asset_name = term.cfg.params["asset_name"]
        asset = base_env.scene[asset_name]

        def read_records():
            return [json.loads(line) for line in output_path.read_text().splitlines()]

        def replay_state(env_ids=None, is_relative=False):
            """Copy selected scene-state rows into the format expected by reset_to."""
            ids = list(range(NUM_ENVS)) if env_ids is None else env_ids.tolist()
            state = base_env.scene.get_state(is_relative=is_relative)
            # reset_to expects only the selected rows, even when env_ids is non-contiguous.
            # This fixture has no surface grippers, whose scene state uses a raw tensor.
            selected_state = {}
            for asset_type, assets in state.items():
                selected_state[asset_type] = {}
                for name, asset_state in assets.items():
                    selected_state[asset_type][name] = {
                        field: values[ids].clone() for field, values in asset_state.items()
                    }
            return selected_state

        # The first reset captures each environment without recording a finished episode.
        if initial_reset_mode == "reset_to":
            # Pin the existing metric restriction independently of the per-env JSONL lifecycle.
            env_ids = torch.tensor([0], device=base_env.device)
            with pytest.raises(AssertionError):
                base_env.reset_to(replay_state(env_ids), env_ids=env_ids)
            assert notifications == [], "SuccessRateMetric must reject the partial reset before JSONL capture"
            state = replay_state()
            state["rigid_object"][asset_name]["root_pose"][:, 2] += 0.3
            base_env.reset_to(state, env_ids=None)
            assert notifications == [[0, 1]], initial_reset_mode
        elif initial_reset_mode == "partial_reset_to":
            env_ids = torch.tensor([0], device=base_env.device)
            base_env.reset_to(replay_state(env_ids), env_ids=env_ids)
            assert notifications == [[0]], initial_reset_mode
            assert output_path.read_text() == ""
            base_env.reset(env_ids=torch.tensor([1], device=base_env.device))
            assert notifications == [[0], [1]], initial_reset_mode
        else:
            env.reset()
            assert notifications == [[0, 1]], initial_reset_mode
        initial_notifications = list(notifications)
        initial_poses = dict(term.starting_poses)
        assert output_path.read_text() == "", "The first reset must not record an episode"
        assert initial_poses == {i: asset.data.root_pose_w.torch[i].tolist() for i in range(NUM_ENVS)}

        if initial_reset_mode != "reset":
            # These cases cover startup only; exercise the shared lifecycle once in the reset case.
            env.reset()
            records = read_records()
            assert [(r["env_id"], r["episode_in_env"]) for r in records] == [(0, 0), (1, 0)], initial_reset_mode
            assert {r["env_id"]: r["starting_pose"] for r in records} == initial_poses, initial_reset_mode
            return True

        # Both live poses and the next reset poses differ from the previous starting state.
        moved_poses = asset.data.root_pose_w.torch.clone()
        moved_poses[:, 0] += 0.1
        asset.write_root_pose_to_sim(moved_poses)
        reset_cfg = base_env.event_manager.get_term_cfg("reset_pick_up_object_pose")
        reset_cfg.params["pose_list"] = [
            Pose(position_xyz=(0.0, -0.5, 0.4), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            Pose(position_xyz=(-0.5, -0.5, 0.4), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        ]
        base_env.episode_length_buf[:] = 17
        base_env.reset(env_ids=torch.tensor([1], device=base_env.device))
        assert notifications == initial_notifications + [[1]], "Partial reset must capture only env 1"
        assert term.starting_poses[0] == initial_poses[0], "Partial reset must preserve env 0's capture"
        assert term.starting_poses[1] == asset.data.root_pose_w.torch[1].tolist()
        assert term.starting_poses[1] != initial_poses[1], "Capture must run after the new reset pose is applied"
        assert base_env.get_episode_index(0) == 0
        assert base_env.get_episode_index(1) == 1
        records = read_records()
        assert len(records) == 1
        assert records[0]["env_id"] == 1
        assert records[0]["episode_in_env"] == records[0]["starting_episode_index"] == 0
        assert records[0]["starting_pose"] == initial_poses[1], "Record the old capture before replacing it"
        assert records[0][CUSTOM_KEY] == 1
        assert records[0]["episode_length"] == 17, "Record the finishing length before reset clears the buffer"

        env.reset()
        assert notifications == initial_notifications + [[1], [0, 1]]
        records = read_records()
        assert [(r["env_id"], r["episode_in_env"]) for r in records] == [(1, 0), (0, 0), (1, 1)]
        assert records[1]["starting_pose"] == initial_poses[0]

        # Replay must capture the supplied state exactly once, including partial/relative restores.
        for env_ids, is_relative in ((None, False), (torch.tensor([1], device=base_env.device), True)):
            ids = list(range(NUM_ENVS)) if env_ids is None else env_ids.tolist()
            phase = f"replay env_ids={ids}, is_relative={is_relative}"
            previous_poses = dict(term.starting_poses)
            state = replay_state(env_ids, is_relative)
            state["rigid_object"][asset_name]["root_pose"][:, 2] += 0.7
            notification_count = len(notifications)
            base_env.reset_to(state, env_ids=env_ids, is_relative=is_relative)
            assert notifications[notification_count:] == [ids], phase
            for env_id in ids:
                assert term.starting_poses[env_id] == asset.data.root_pose_w.torch[env_id].tolist(), phase
                assert term.starting_poses[env_id] != previous_poses[env_id], phase
            if env_ids is not None:
                assert term.starting_poses[0] == previous_poses[0], phase
            replayed_poses = dict(term.starting_poses)
            env.reset()
            records = read_records()
            assert {r["env_id"]: r["starting_pose"] for r in records[-NUM_ENVS:]} == replayed_poses, phase

        def exported_demos():
            recorder = base_env.recorder_manager
            return recorder.exported_successful_episode_count + recorder.exported_failed_episode_count

        # Failed restores never initialize a new episode or reuse its stale capture on retry.
        for failure_stage in ("event", "scene", "observation", "term"):
            env_ids = torch.tensor([1], device=base_env.device)
            state = replay_state(env_ids)
            state["rigid_object"][asset_name]["root_pose"][:, 2] += 0.7
            records_before = len(records)
            exports_before = exported_demos()
            previous_index = base_env.get_episode_index(1)
            notification_count = len(notifications)
            if failure_stage == "event":
                failure = patch.object(reset_cfg, "func", side_effect=ValueError("placement reset failed"))
                error_type = ValueError
            elif failure_stage == "scene":
                del state["rigid_object"][asset_name]
                failure = nullcontext()
                error_type = KeyError
            elif failure_stage == "observation":
                failure = patch.object(
                    base_env.observation_manager, "compute", side_effect=ValueError("observation failed")
                )
                error_type = ValueError
            else:
                failure = patch.object(term, "reset", side_effect=ValueError("capture failed"))
                error_type = RuntimeError
            with failure, pytest.raises(error_type) as error:
                base_env.reset_to(state, env_ids=env_ids)
            if failure_stage == "term":
                assert "starting_pose" in str(error.value)
                assert isinstance(error.value.__cause__, ValueError)
            assert notifications[notification_count:] == [], failure_stage
            records = read_records()
            assert len(records) == records_before + 1, failure_stage
            assert exported_demos() == exports_before + 1, failure_stage
            assert records[-1]["episode_in_env"] == previous_index, failure_stage

            # Retrying both envs finishes only the unaffected episode, then captures both starts.
            env.reset()
            assert notifications[notification_count:] == [[0, 1]], failure_stage
            records = read_records()
            assert len(records) == records_before + 2, failure_stage
            # Isaac Lab exports both reset attempts; only the valid one has an Arena JSONL row.
            assert exported_demos() == exports_before + 3, failure_stage
            assert records[-1]["env_id"] == 0, failure_stage
            assert base_env.get_episode_index(1) == previous_index + 2, failure_stage
            env.reset()
            records = read_records()
            assert records[-1]["episode_in_env"] == previous_index + 2, failure_stage

        # A write can fail after one environment's row is persisted. Retrying must not duplicate it.
        records_before = len(records)
        notification_count = len(notifications)
        previous_indices = [base_env.get_episode_index(i) for i in range(NUM_ENVS)]
        append_record = base_env.episode_recorder_manager._append_record

        def append_then_fail(record):
            append_record(record)
            raise OSError("episode output failed after append")

        with patch.object(base_env.episode_recorder_manager, "_append_record", side_effect=append_then_fail):
            with pytest.raises(OSError, match="episode output failed after append"):
                env.reset()
        records = read_records()
        assert len(records) == records_before + 1, "The first row persists even though the batch failed"
        assert records[-1]["env_id"] == 0, "The second environment's row must not have been written"
        assert notifications[notification_count:] == [], "Failed recording must stop the reset before capture"
        assert [base_env.get_episode_index(i) for i in range(NUM_ENVS)] == previous_indices

        env.reset()
        assert read_records() == records, "Retry must not re-emit any finishing episode from the failed batch"
        assert notifications[notification_count:] == [[0, 1]], "Retry must capture fresh starts for both environments"
        env.reset()
        records = read_records()
        assert len(records) == records_before + 3, "Both fresh episodes must be recordable after recovery"
        assert [(record["env_id"], record["episode_in_env"]) for record in records[-NUM_ENVS:]] == [
            (env_id, episode_index + 1) for env_id, episode_index in enumerate(previous_indices)
        ], "Failure before reset events must not reserve an extra episode index"

        # Force a timeout through step() to exercise automatic resets as well as explicit ones.
        previous_poses = dict(term.starting_poses)
        notification_count = len(notifications)
        base_env.episode_length_buf[:] = base_env.max_episode_length - 1
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=base_env.device))
        assert notifications[notification_count:] == [[0, 1]], "Timeout reset must capture both environments once"
        records = read_records()
        assert {r["env_id"]: r["starting_pose"] for r in records[-NUM_ENVS:]} == previous_poses

        # Check episode identity and variation attribution across every reset path above.
        variation_key = f"{asset_name}.{VARIATION_NAME}"
        last_episode_by_env = {}
        for record in records:
            assert record["episode_in_env"] > last_episode_by_env.get(record["env_id"], -1), record
            last_episode_by_env[record["env_id"]] = record["episode_in_env"]
            assert record["starting_episode_index"] == record["episode_in_env"], record
            assert record["variations"][variation_key] == VARIATION_SAMPLE, record
        assert len(reset_lengths) == sum(len(ids) for ids in notifications), "Capture every notified environment"
        assert all(length == 0 for _, _, length in reset_lengths), reset_lengths
    finally:
        env.close()
    return True


def _nested_recorder_cfg(leaf_cfg, outer_name="subtask_0", inner_name="subtask_1"):
    """Nest a leaf with configurable term names and fixed JSON output namespaces."""
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg, NamespacedEpisodeRecorder

    inner = EpisodeRecorderTermCfg(
        func=NamespacedEpisodeRecorder, params={"namespace": "subtask_1", "terms": {"capture": leaf_cfg}}
    )
    return SimpleNamespace(**{
        outer_name: EpisodeRecorderTermCfg(
            func=NamespacedEpisodeRecorder, params={"namespace": "subtask_0", "terms": {inner_name: inner}}
        )
    })


@pytest.mark.parametrize("term_names", [("subtask_0", "subtask_1"), ("task_recorders", "child_recorders")])
def test_nested_episode_recorder_validation(term_names):
    """Nested configuration validation needs no SimulationApp or scene."""
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg

    def record_static(env, env_id, field_name):
        return {field_name: 1}

    env = SimpleNamespace(num_envs=1, sim=SimpleNamespace(is_playing=lambda: True))
    for leaf_cfg, error_type in (
        (EpisodeRecorderTermCfg(func=record_static, params={"field": 1}), ValueError),
        (EpisodeRecorderTermCfg(func=123), AttributeError),
        (EpisodeRecorderTermCfg(func=object), TypeError),
        (object(), TypeError),
    ):
        with pytest.raises(error_type, match=f"{term_names[0]}.*{term_names[1]}.*capture"):
            EpisodeRecorderManager(_nested_recorder_cfg(leaf_cfg, *term_names), env)


@pytest.mark.parametrize("term_names", [("subtask_0", "subtask_1"), ("task_recorders", "child_recorders")])
def test_nested_episode_recorder_diagnostics(tmp_path, term_names):
    """Namespaced failures identify the leaf term without starting a simulation."""
    from isaaclab.managers import ManagerTermBase

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg

    instances = []

    class BrokenTerm(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            instances.append(self)

        def reset(self, env_ids=None):
            raise ValueError("capture failed")

        def __call__(self, env, env_id):
            return {"unserializable": object()}

    env = SimpleNamespace(num_envs=1, sim=SimpleNamespace(is_playing=lambda: True))
    manager = EpisodeRecorderManager(_nested_recorder_cfg(EpisodeRecorderTermCfg(func=BrokenTerm), *term_names), env)
    assert len(instances) == 1, "Recursive validation must not instantiate child terms twice"
    path = f"{term_names[0]}/{term_names[1]}/capture"
    with pytest.raises(RuntimeError, match=path) as error:
        manager.reset([0])
    assert isinstance(error.value.__cause__, ValueError), "Preserve the original leaf failure"
    with pytest.raises(TypeError, match=f"{path}.*non-JSON-serializable"):
        manager.record_pre_reset([0])

    output_path = tmp_path / "namespaced_episodes.jsonl"
    manager.set_output_path(output_path)
    with patch.object(BrokenTerm, "__call__", return_value={"value": 42}):
        manager.record_pre_reset([0])
    assert json.loads(output_path.read_text()) == {
        "job_name": "default",
        "subtask_0": {"subtask_1": {"value": 42}},
    }, "Diagnostic term names must not change JSON output namespaces"


def test_episode_recorder_reset_contract():
    """Stateful reset dispatch needs only a fake environment, not SimulationApp."""
    from isaaclab.managers import ManagerTermBase

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg

    received_ids = []

    class StatefulTerm(ManagerTermBase):
        def reset(self, env_ids=None):
            received_ids.append(env_ids)

        def __call__(self, env, env_id):
            return {}

    class FailingTerm(StatefulTerm):
        def reset(self, env_ids=None):
            raise ValueError("cannot capture starting state")

    env = SimpleNamespace(num_envs=2, sim=SimpleNamespace(is_playing=lambda: True))
    manager = EpisodeRecorderManager(
        SimpleNamespace(
            stateful=EpisodeRecorderTermCfg(func=StatefulTerm),
            plain=EpisodeRecorderTermCfg(func=record_step_bucket),
        ),
        env,
    )
    for index, env_ids in enumerate((torch.tensor([1]), (1, 0), None, [])):
        assert manager.reset(env_ids) == {}, "Return an empty logging dictionary"
        assert len(received_ids) == index + 1, "Notify each stateful term exactly once per reset"
        assert received_ids[-1] is env_ids, "Pass environment IDs through without changing their representation"
    failing_manager = EpisodeRecorderManager(SimpleNamespace(broken=EpisodeRecorderTermCfg(func=FailingTerm)), env)
    with pytest.raises(RuntimeError, match="Episode recorder term 'broken' failed during reset") as error:
        failing_manager.reset()
    assert isinstance(error.value.__cause__, ValueError)


def _test_builder_episode_recorder_terms(simulation_app):  # noqa: ARG001
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.tasks.no_task import NoTask

    environment_term = EpisodeRecorderTermCfg(func=record_step_bucket)
    task_term = EpisodeRecorderTermCfg(func=record_step_bucket)
    arena_env = SimpleNamespace(episode_recorder_terms={"environment": environment_term})
    task_terms = {"task": task_term}

    def get_terms(received_env):
        assert received_env is arena_env
        return task_terms

    task = SimpleNamespace(get_episode_recorder_terms=get_terms)
    builder = ArenaEnvBuilder.__new__(ArenaEnvBuilder)
    builder.arena_env = arena_env
    assert NoTask().get_episode_recorder_terms(arena_env) == {}
    terms = builder._collect_episode_recorder_terms(task)
    assert terms == {"environment": environment_term, "task": task_term}
    assert arena_env.episode_recorder_terms == {"environment": environment_term}
    assert task_terms == {"task": task_term}
    cfg = builder._compose_episode_recorders_cfg(terms)
    assert set(vars(cfg)) == {"core", "variations", "progress", "environment", "task"}

    task_terms["environment"] = task_term
    with pytest.raises(AssertionError, match="contributed by both the environment and task"):
        builder._collect_episode_recorder_terms(task)
    del task_terms["environment"]
    for reserved_name in ("core", "variations", "progress"):
        for contributor in (arena_env.episode_recorder_terms, task_terms):
            contributor[reserved_name] = task_term
            with pytest.raises(AssertionError, match="collides with a built-in term"):
                builder._compose_episode_recorders_cfg(builder._collect_episode_recorder_terms(task))
            del contributor[reserved_name]
    return True


def _test_composite_episode_recorder_terms(simulation_app, output_dir):  # noqa: ARG001
    from isaaclab.managers import ManagerTermBase

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.utils.configclass import make_configclass

    arena_env = object()
    notifications = []

    class Capture(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env_ids = set()

        def reset(self, env_ids=None):
            ids = range(self.num_envs) if env_ids is None else env_ids
            self.env_ids.update(int(i) for i in ids)
            notifications.append((self.cfg.params["value"], list(ids)))

        def __call__(self, env, env_id, value):
            assert env_id in self.env_ids
            return {"value": value, "captured_env": env_id}

    def record_static(env, env_id, field_name):
        return {field_name: 42}

    class RecordingTask(NoTask):
        def __init__(self, value, field_name="static"):
            super().__init__()
            self.value = value
            self.field_name = field_name

        def get_episode_recorder_terms(self, received_env):
            assert received_env is arena_env
            return {
                "capture": EpisodeRecorderTermCfg(func=Capture, params={"value": self.value}),
                "static": EpisodeRecorderTermCfg(func=record_static, params={"field_name": self.field_name}),
            }

    env = SimpleNamespace(num_envs=2, sim=SimpleNamespace(is_playing=lambda: True))
    for composite_type in (CompositeTaskBase, SequentialTaskBase):
        task = composite_type([
            RecordingTask(10),
            RecordingTask(20),
            CompositeTaskBase([NoTask(), RecordingTask(30)]),
        ])
        terms = task.get_episode_recorder_terms(arena_env)
        assert set(terms) == {"subtask_0", "subtask_1", "subtask_2"}
        cfg = make_configclass(
            "SubtaskRecordersCfg", [(name, EpisodeRecorderTermCfg, term) for name, term in terms.items()]
        )()
        manager = EpisodeRecorderManager(cfg, env)
        output_path = Path(output_dir) / "composite_episodes.jsonl"
        manager.set_output_path(output_path)
        notifications.clear()
        manager.reset(torch.tensor([0]))
        manager.reset(torch.tensor([1]))
        assert notifications == [(10, [0]), (20, [0]), (30, [0]), (10, [1]), (20, [1]), (30, [1])]
        manager.record_pre_reset([0, 1])
        records = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert records == [
            {
                "job_name": "default",
                "subtask_0": {"value": 10, "captured_env": i, "static": 42},
                "subtask_1": {"value": 20, "captured_env": i, "static": 42},
                "subtask_2": {"subtask_1": {"value": 30, "captured_env": i, "static": 42}},
            }
            for i in range(2)
        ]
        assert terms["subtask_0"].params["terms"]["capture"].func is Capture

        # Namespacing separates children but must not hide collisions within one child.
        task = composite_type([RecordingTask(10, field_name="value")])
        manager = EpisodeRecorderManager(SimpleNamespace(**task.get_episode_recorder_terms(arena_env)), env)
        manager.reset([0])
        with pytest.raises(AssertionError, match="subtask_0/static.*redefines fields"):
            manager.record_pre_reset([0])
    return True


def test_composite_episode_recorder_terms(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_composite_episode_recorder_terms, headless=HEADLESS, output_dir=tmp_path
    ), "composite episode recorder terms test failed"


@pytest.mark.parametrize("initial_reset_mode", ["reset", "reset_to", "partial_reset_to"])
def test_task_recorder_lifecycle(tmp_path, initial_reset_mode):
    assert run_function_with_persistent_simulation_app(
        _test_task_recorder_lifecycle, headless=HEADLESS, output_dir=tmp_path, initial_reset_mode=initial_reset_mode
    ), "task recorder lifecycle test failed"


def test_builder_episode_recorder_terms():
    assert run_function_with_persistent_simulation_app(
        _test_builder_episode_recorder_terms, headless=HEADLESS
    ), "builder episode recorder terms test failed"


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
        for initial_reset_mode in ("reset", "reset_to", "partial_reset_to"):
            test_task_recorder_lifecycle(Path(_tmp_dir), initial_reset_mode=initial_reset_mode)
        test_builder_episode_recorder_terms()
        test_episode_recorder_reset_contract()
        for term_names in (("subtask_0", "subtask_1"), ("task_recorders", "child_recorders")):
            test_nested_episode_recorder_validation(term_names)
            test_nested_episode_recorder_diagnostics(Path(_tmp_dir), term_names)
        test_composite_episode_recorder_terms(Path(_tmp_dir))
