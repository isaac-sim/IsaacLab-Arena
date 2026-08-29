# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import torch
import tqdm
from dataclasses import field
from pathlib import Path

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


def _get_parent_transform_diagnostics(frame_view) -> tuple[list[list[float]], list[list[float]], int]:
    """Read parent world positions from Fabric and USD without creating another FrameView."""
    import warp as wp
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab.utils.warp import fabric as fabric_utils
    from pxr import Usd, UsdGeom

    parent_positions = wp.zeros((frame_view.count, 3), dtype=wp.float32, device=frame_view.device)
    parent_orientations = wp.zeros((frame_view.count, 4), dtype=wp.float32, device=frame_view.device)
    wp.launch(
        kernel=fabric_utils.decompose_indexed_fabric_transforms,
        dim=frame_view.count,
        inputs=[
            frame_view._get_parent_world_ifa(),
            parent_positions,
            parent_orientations,
            frame_view._fabric_empty_2d_array_sentinel,
            frame_view._view_indices,
        ],
        device=frame_view.device,
    )
    wp.synchronize()

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_parent_positions = []
    for prim in frame_view._usd_view.prims:
        translation = xform_cache.GetLocalToWorldTransform(prim.GetParent()).ExtractTranslation()
        usd_parent_positions.append([float(translation[0]), float(translation[1]), float(translation[2])])

    return wp.to_torch(parent_positions).tolist(), usd_parent_positions, get_current_stage_id()


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

    env 0's box lands in the drawer (success) while env 1's box lands outside it (failure).

    Args:
        output_dir: Directory the JSONL records are written into.
        episode_recorder_terms: Extra per-episode recorder terms (i.e. EpisodeRecorderTermCfg.
        enable_variation: When True, attach an enabled run-time variation to the cracker box.

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
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
        episode_recorder_terms=episode_recorder_terms or {},
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    args_cli.disable_fabric = os.environ.get("ISAACLAB_ARENA_DIAGNOSTIC_DISABLE_FABRIC") == "1"
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
    env.reset()
    return env, output_path


def _log_frame_view_hierarchy_boundary(env, boundary: str) -> None:
    """Log child and parent transforms after FrameView initialization."""
    import omni.kit.app

    success_term = env.unwrapped.termination_manager.get_term_cfg("success").func
    frame_view = success_term._destination_asset_base_pose_reader._frame_view
    assert frame_view._fabric_initialized, "Initialize FrameView before taking an observation-only snapshot."
    hierarchy = frame_view._fabric_hierarchy
    tracking_local_before_reads = hierarchy.tracking_local_xform_changes
    tracking_world_before_reads = hierarchy.tracking_world_xform_changes
    fabric_world_position, fabric_world_orientation = frame_view.get_world_poses()
    fabric_local_position, fabric_local_orientation = frame_view.get_local_poses()
    usd_world_position, usd_world_orientation = frame_view._usd_view.get_world_poses()
    usd_local_position, usd_local_orientation = frame_view._usd_view.get_local_poses()
    fabric_parent_position, usd_parent_position, active_stage_id = _get_parent_transform_diagnostics(frame_view)
    tracking_local_after_reads = hierarchy.tracking_local_xform_changes
    tracking_world_after_reads = hierarchy.tracking_world_xform_changes
    parent_paths = [path.rsplit("/", 1)[0] for path in frame_view.prim_paths]
    physx_ui_enabled = omni.kit.app.get_app().get_extension_manager().is_extension_enabled("omni.physx.ui")
    print(
        f"[hierarchy-boundary] boundary={boundary} "
        f"physx_ui_enabled={physx_ui_enabled} "
        f"children={frame_view.prim_paths} "
        f"parents={parent_paths} "
        f"child_fabric_world_position={fabric_world_position.torch.tolist()} "
        f"child_fabric_world_orientation={fabric_world_orientation.torch.tolist()} "
        f"child_fabric_local_position={fabric_local_position.torch.tolist()} "
        f"child_fabric_local_orientation={fabric_local_orientation.torch.tolist()} "
        f"child_usd_world_position={usd_world_position.torch.tolist()} "
        f"child_usd_world_orientation={usd_world_orientation.torch.tolist()} "
        f"child_usd_local_position={usd_local_position.torch.tolist()} "
        f"child_usd_local_orientation={usd_local_orientation.torch.tolist()} "
        f"parent_fabric_world_position={fabric_parent_position} "
        f"parent_usd_world_position={usd_parent_position} "
        f"tracking_local_before_reads={tracking_local_before_reads} "
        f"tracking_local_after_reads={tracking_local_after_reads} "
        f"tracking_world_before_reads={tracking_world_before_reads} "
        f"tracking_world_after_reads={tracking_world_after_reads} "
        f"fabric_id={frame_view._fabric_id} "
        f"active_stage_id={active_stage_id} "
        f"usdrt_stage_id={frame_view._stage.GetStageIdAsStageId()}",
        flush=True,
    )


def _test_frame_view_reconciliation_boundary(simulation_app, output_dir):  # noqa: ARG001
    env, _ = create_recorder_env(output_dir)
    originals = []
    step_number = 0

    def observe_after(obj, method_name: str, boundary: str) -> None:
        original = getattr(obj, method_name)
        originals.append((obj, method_name, original))

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            _log_frame_view_hierarchy_boundary(env, f"step-{step_number}-{boundary}")
            return result

        setattr(obj, method_name, observed)

    try:
        success_term = env.unwrapped.termination_manager.get_term_cfg("success").func
        frame_view = success_term._destination_asset_base_pose_reader._frame_view
        # The pinned FrameView lazily initializes on its first getter and seeds
        # child and parent Fabric matrices from USD. Keep that write in setup so
        # every logged boundary below is observation-only.
        frame_view.get_world_poses()
        _log_frame_view_hierarchy_boundary(env, "initial-after-frame-view-initialization")
        observe_after(env.unwrapped.action_manager, "apply_action", "after-apply-action")
        observe_after(env.unwrapped.scene, "write_data_to_sim", "after-scene-write")
        observe_after(env.unwrapped.sim, "step", "after-physics")
        observe_after(env.unwrapped.sim, "render", "after-render")
        observe_after(env.unwrapped.scene, "update", "after-scene-update")
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for step in range(1, 3):
            step_number = step
            env.step(actions)
            _log_frame_view_hierarchy_boundary(env, f"step-{step}-after-env-step")
    finally:
        for obj, method_name, original in reversed(originals):
            setattr(obj, method_name, original)
        env.close()
    return True


def test_frame_view_reconciliation_boundary(tmp_path):
    """Locate which environment-step phase reconciles Fabric world transforms."""
    from isaaclab.utils.warp import fabric as fabric_utils

    if not hasattr(fabric_utils, "decompose_indexed_fabric_transforms"):
        pytest.skip("The installed Isaac Lab predates indexed Fabric transform diagnostics")
    assert run_function_with_persistent_simulation_app(
        _test_frame_view_reconciliation_boundary,
        headless=HEADLESS,
        output_dir=tmp_path,
    )


def _roll_out_and_read_episode_record(env, output_path) -> list[dict]:
    """Step the env for ``NUM_STEPS`` (records stream to disk as episodes finish), then parse them."""
    log_state = os.environ.get("ISAACLAB_ARENA_DIAGNOSTIC_LOG_STATE") == "1"
    max_force_by_env = torch.zeros(NUM_ENVS, device=env.unwrapped.device)
    for step in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            if log_state:
                sensor = env.unwrapped.scene.sensors["contact_sensor_cracker_box"]
                force_matrix = sensor.data.force_matrix_w.torch
                force_by_env = torch.linalg.vector_norm(force_matrix, dim=-1).flatten(start_dim=1).amax(dim=1)
                max_force_by_env = torch.maximum(max_force_by_env, force_by_env)
                if step < 5 or (step + 1) % 20 == 0:
                    cracker_box = env.unwrapped.scene["cracker_box"]
                    success = env.unwrapped.termination_manager.get_term("success")
                    success_term = env.unwrapped.termination_manager.get_term_cfg("success").func
                    destination_reader = success_term._destination_asset_base_pose_reader
                    frame_view = destination_reader._frame_view
                    destination_pose = destination_reader.get_pose_in_world_frame()
                    fabric_local_position, fabric_local_orientation = frame_view.get_local_poses()
                    usd_position, usd_orientation = frame_view._usd_view.get_world_poses()
                    usd_local_position, usd_local_orientation = frame_view._usd_view.get_local_poses()
                    fabric_parent_position, usd_parent_position, active_stage_id = _get_parent_transform_diagnostics(
                        frame_view
                    )
                    hierarchy = frame_view._fabric_hierarchy
                    if step == 0:
                        parent_paths = [path.rsplit("/", 1)[0] for path in frame_view.prim_paths]
                        print(
                            f"[frame-view-layout] children={frame_view.prim_paths} "
                            f"parents={parent_paths} "
                            f"fabric_id={frame_view._fabric_id} "
                            f"active_stage_id={active_stage_id} "
                            f"usdrt_stage_id={frame_view._stage.GetStageIdAsStageId()}",
                            flush=True,
                        )
                    print(
                        "[contact-state] "
                        f"step={step + 1} "
                        f"force={force_by_env.tolist()} "
                        f"max_force={max_force_by_env.tolist()} "
                        f"success={success.tolist()} "
                        f"position={cracker_box.data.root_pos_w.torch.tolist()} "
                        f"velocity={cracker_box.data.root_lin_vel_w.torch.tolist()}",
                        f"destination_fabric={destination_pose.tolist()} "
                        f"destination_usd_position={usd_position.torch.tolist()} "
                        f"destination_usd_orientation={usd_orientation.torch.tolist()} "
                        f"destination_fabric_local_position={fabric_local_position.torch.tolist()} "
                        f"destination_fabric_local_orientation={fabric_local_orientation.torch.tolist()} "
                        f"destination_usd_local_position={usd_local_position.torch.tolist()} "
                        f"destination_usd_local_orientation={usd_local_orientation.torch.tolist()} "
                        f"parent_fabric_world_position={fabric_parent_position} "
                        f"parent_usd_world_position={usd_parent_position} "
                        f"tracking_local={hierarchy.tracking_local_xform_changes} "
                        f"tracking_world={hierarchy.tracking_world_xform_changes} "
                        f"active_stage_id={active_stage_id}",
                        flush=True,
                    )
                    if os.environ.get("ISAACLAB_ARENA_DIAGNOSTIC_ASSERT_FABRIC_STABLE") == "1":
                        fabric_world_position = destination_pose[:, :3]
                        usd_world_position = usd_position.torch
                        assert torch.allclose(fabric_world_position, usd_world_position, atol=1e-5, rtol=1e-5), (
                            f"Fabric world pose diverged from USD at step {step + 1}: "
                            f"fabric_world={fabric_world_position.tolist()}, "
                            f"fabric_local={fabric_local_position.torch.tolist()}, "
                            f"usd_world={usd_world_position.tolist()}"
                        )

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
