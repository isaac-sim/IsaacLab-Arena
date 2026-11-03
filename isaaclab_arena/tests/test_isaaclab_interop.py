# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test deferred Gym construction and Arena/Isaac Lab CLI argument ownership."""

import gc
import gymnasium as gym
import json
import sys
import torch
import weakref
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _RecordingEnv(gym.Env):
    """Replace simulation construction while keeping real variation listeners and Gym wrapping."""

    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Discrete(1)

    def __init__(
        self, cfg, variation_recorder=None, variation_replay_scheduler=None, constructor_option=None, **kwargs
    ):
        self.cfg = cfg
        self.variation_recorder = variation_recorder
        self.variation_replay_scheduler = variation_replay_scheduler
        self.constructor_option = constructor_option
        self.episode_index = -1
        if variation_recorder is not None:
            variation_recorder.bind_env(self)

    def get_episode_index(self, env_id):
        return self.episode_index

    def reset(self, **kwargs):
        if self.episode_index >= 0:
            self.variation_replay_scheduler.complete_episodes([0])
        self.episode_index += 1
        self.variation_replay_scheduler.assign_new_episodes([0])
        sample = self.cfg.events.mass.params["sampler"].sample(1, env_ids=torch.tensor([0]))
        return {}, {"sample": sample}


def test_rl_discovery_placeholder_declares_automatic_registration():
    from isaaclab_tasks.utils.task_registration import TASK_REGISTRATION_CALLBACK_KEY

    from isaaclab_arena.environments import isaaclab_interop

    task_name = "arena_interop_discovery_test"
    gym.registry.pop(task_name, None)
    try:
        isaaclab_interop.register_rl_discovery_placeholder(task_name)

        spec = gym.spec(task_name)
        assert spec.kwargs["_arena_rl_discovery_placeholder"]
        assert spec.kwargs["default_agent"] == "rsl_rl"
        assert spec.kwargs[TASK_REGISTRATION_CALLBACK_KEY].endswith(".environment_registration_callback")
    finally:
        gym.registry.pop(task_name, None)


def test_arena_task_discovery_is_idempotent_and_preserves_native_tasks(monkeypatch):
    from isaaclab_arena.environments import isaaclab_interop

    arena_task = "arena_interop_catalogue_test"
    native_task = "arena_interop_native_collision_test"
    gym.registry.pop(arena_task, None)
    gym.registry.pop(native_task, None)
    gym.register(native_task, entry_point=lambda: None, kwargs={"native": True})
    monkeypatch.setattr(isaaclab_interop, "ensure_environments_registered", lambda: None)
    monkeypatch.setattr(
        isaaclab_interop.EnvironmentRegistry,
        "get_all_keys",
        lambda _registry: [arena_task, native_task],
    )
    try:
        isaaclab_interop.register_arena_tasks_for_discovery()
        isaaclab_interop.register_arena_tasks_for_discovery()

        assert gym.spec(arena_task).kwargs["_arena_rl_discovery_placeholder"]
        assert gym.spec(native_task).kwargs == {"native": True}
    finally:
        gym.registry.pop(arena_task, None)
        gym.registry.pop(native_task, None)


def _test_registered_environment_preserves_runtime_recorders(
    _simulation_app, monkeypatch, tmp_path, entry_point_is_string
):
    """Gym-only callers retain listener identity without keeping a released env alive."""
    from isaaclab_arena.environments import arena_env_builder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation
    from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay
    from isaaclab_arena.variations.variation_recorder import VariationRecorder

    variation = ObjectMassVariation("asset")
    variation.enable()
    variations = {"asset": [variation]}
    replay_path = tmp_path / "variations.jsonl"
    replay_path.write_text(
        "".join(json.dumps({"variations": {"asset.mass": [value]}}) + "\n" for value in (0.1, 0.2)),
        encoding="utf-8",
    )
    scheduler = configure_recorded_variation_replay(str(replay_path), variations)
    recorder = VariationRecorder()
    recorder.attach(variations)
    _, mass_event = variation.build_event_cfg()
    cfg = SimpleNamespace(events=SimpleNamespace(mass=mass_event), sim=SimpleNamespace(), demo_recorder_config=None)
    arena_env = SimpleNamespace(
        name="arena_interop_recorders_test",
        scene=SimpleNamespace(validate_simulation_cfg=lambda cfg: None),
        rl_framework_entry_point=None,
    )
    builder = arena_env_builder.ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(device="cpu"))
    entry_point = f"{__name__}:_RecordingEnv" if entry_point_is_string else _RecordingEnv
    monkeypatch.setattr(builder, "get_entry_point", lambda: entry_point)
    monkeypatch.setattr(arena_env_builder, "install_empty_cpu_warp_to_torch_patch", lambda: None)
    monkeypatch.setattr(arena_env_builder, "apply_arena_global_settings", lambda: None)
    monkeypatch.setattr(arena_env_builder, "_configure_arena_visualizer_defaults", lambda cfg: None)
    monkeypatch.setattr(arena_env_builder, "parse_env_cfg", lambda *args, **kwargs: cfg)
    env_kwargs = {
        "variation_recorder": recorder,
        "variation_replay_scheduler": scheduler,
        "constructor_option": "registered",
    }
    try:
        name, registered_cfg, _ = builder.build_registered(cfg, env_kwargs)
        # Isaac Lab loads/modifies cfg separately and passes no Arena constructor kwargs.
        env = gym.make(name, cfg=deepcopy(registered_cfg), constructor_option="explicit")
        try:
            assert env.unwrapped.variation_recorder is recorder
            assert env.unwrapped.variation_replay_scheduler is scheduler
            assert env.unwrapped.constructor_option == "explicit"
            for episode_index, expected in enumerate((0.1, 0.2)):
                _, info = env.reset()
                assert info["sample"].item() == pytest.approx(expected)
                recorded = recorder["asset.mass"].sample_for_episode(0, episode_index)
                assert recorded.item() == pytest.approx(expected)
            env_ref = weakref.ref(env.unwrapped)
        finally:
            env.close()
        del env
        gc.collect()
        assert name in gym.registry
        assert env_ref() is None
    finally:
        gym.registry.pop(arena_env.name, None)
    return True


@pytest.mark.parametrize("entry_point_is_string", [True, False], ids=["string", "class"])
def test_registered_environment_preserves_runtime_recorders(monkeypatch, tmp_path, entry_point_is_string):
    assert run_function_with_persistent_simulation_app(
        _test_registered_environment_preserves_runtime_recorders,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        entry_point_is_string=entry_point_is_string,
    )


def _test_registration_callback_routes_variation_overrides(_simulation_app, monkeypatch, variation_host):
    """Apply Arena variations while leaving Isaac Lab CLI and Hydra options untouched."""
    from isaaclab_arena.environments import arena_env_builder, isaaclab_interop
    from isaaclab_arena.variations import variations_hydra
    from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation

    variation = ObjectMassVariation(variation_host)
    # Asset names may match Lab's config roots, including assets without variations.
    variations = {"env": [], "agent": [], variation_host: [variation]}
    arena_env = SimpleNamespace(scene=SimpleNamespace(get_asset_variations=lambda: variations), embodiment=None)
    task_name = "arena_interop_variations_test"
    factory = object()
    monkeypatch.setitem(isaaclab_interop.EnvironmentRegistry()._components, task_name, factory)
    monkeypatch.setattr(isaaclab_interop, "is_simulation_app_running", lambda: True)
    monkeypatch.setattr(isaaclab_interop, "_purge_leaked_isaaclab_assets_presets", lambda: None)
    monkeypatch.setattr(isaaclab_interop, "ensure_environments_registered", lambda: None)
    monkeypatch.setattr(isaaclab_interop, "add_environment_cli_args", lambda parser, factory: None)
    monkeypatch.setattr(isaaclab_interop, "build_environment_from_cli", lambda factory, args: arena_env)
    received_overrides = []

    def build_registered(builder):
        received_overrides.extend(builder.hydra_overrides or [])
        variations_hydra.apply_overrides(builder.get_all_variations(), builder.hydra_overrides or [])

    monkeypatch.setattr(arena_env_builder.ArenaEnvBuilder, "build_registered", build_registered)
    variation_args = [f"{variation_host}.mass.enabled=true", f"{variation_host}.mass.sampler_cfg.low=[0.2]"]
    lab_args = ["--dataset_file", "demos.hdf5", "env.seed=17", "agent.policy.activation=relu"]
    monkeypatch.setattr(
        sys, "argv", ["record_demos.py", "--task", task_name, "--device", "cpu", *variation_args, *lab_args]
    )

    remaining_args = isaaclab_interop.environment_registration_callback()

    assert received_overrides == variation_args
    assert variation.enabled
    assert variation.cfg.sampler_cfg.low == [0.2]
    assert remaining_args == lab_args
    return True


@pytest.mark.parametrize("variation_host", ["asset", "env", "agent"])
def test_registration_callback_routes_variation_overrides(monkeypatch, variation_host):
    assert run_function_with_persistent_simulation_app(
        _test_registration_callback_routes_variation_overrides, monkeypatch=monkeypatch, variation_host=variation_host
    )


def _test_registered_environment_restarts_recording(_simulation_app, tmp_path, replay):
    """Recreating a registered scene starts fresh recording and replay, including partial resets."""
    import warp as wp

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.pose import Pose

    source = Path(__file__).parent / "test_data/placement_replay.yaml"
    rows = [json.loads(line) for line in source.with_suffix(".jsonl").read_text().splitlines()]
    for index, row in enumerate(rows):
        row["variations"] = {"cube_0.mass": [0.1 * (index + 1)]}
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    builder = ArenaEnvBuilder(
        ArenaEnvGraphSpec.from_yaml(source).to_arena_env(),
        ArenaEnvBuilderCfg(
            num_envs=2,
            device="cpu",
            recorded_variation_samples_path=str(replay_path) if replay else None,
        ),
        hydra_overrides=["cube_0.mass.enabled=true"],
    )

    def check_episode(base, env_id, source_index):
        episode = base.get_episode_index(env_id)
        recorder = base.variation_recorder
        mass = wp.to_torch(base.scene["cube_0"].data.body_mass)[env_id, 0].item()
        assert recorder["cube_0.mass"].sample_for_episode(env_id, episode).item() == pytest.approx(mass)
        placement = recorder.placement_record.sample_for_episode(env_id, episode)
        for key, pose in placement["poses"].items():
            expected = Pose.from_dict(pose).to_tensor(base.device)
            torch.testing.assert_close(base.arena_world.get_pose_e(key)[env_id], expected, atol=2e-5, rtol=0)
        if replay:
            assert mass == pytest.approx(rows[source_index]["variations"]["cube_0.mass"][0])
            assert placement == rows[source_index]["placement"]

    name, cfg, _ = builder.build_registered()
    try:
        for _ in range(2):
            env = gym.make(name, cfg=deepcopy(cfg))
            try:
                base = env.unwrapped
                env.reset()
                check_episode(base, env_id=0, source_index=0)
                check_episode(base, env_id=1, source_index=1)
                base.reset(env_ids=torch.tensor([1], device=base.device))
                check_episode(base, env_id=1, source_index=2)
                assert base.get_episode_index(0) == 0
                assert base.get_episode_index(1) == 1
            finally:
                # Close with both episodes active to exercise reuse of the registered state.
                env.close()
    finally:
        gym.registry.pop(name, None)
    return True


@pytest.mark.parametrize("replay", [False, True], ids=["live", "replay"])
def test_registered_environment_restarts_recording(tmp_path, replay):
    assert run_function_with_persistent_simulation_app(
        _test_registered_environment_restarts_recording, tmp_path=tmp_path, replay=replay
    )
