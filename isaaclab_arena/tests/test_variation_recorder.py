# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for sample listeners and ``VariationRecorder``.

Listener and recorder mechanics are plain Python (no SimulationApp). The end-to-end test that
records a real HDR draw runs inside ``run_function_with_persistent_simulation_app`` because constructing a dome
light via the registry pulls in ``isaaclab.sim``.
"""

import torch
from dataclasses import field

import pytest
from isaaclab.utils.configclass import configclass

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import (
    BuildTimeVariationBase,
    RunTimeVariationBase,
    VariationBaseCfg,
    VariationBuildContext,
)
from isaaclab_arena.variations.variation_recorder import VariationRecord, VariationRecorder

HEADLESS = True


@configclass
class _RecorderTestVariationCfg(VariationBaseCfg):
    """Variation configuration whose sampler draws a scalar per sample."""

    __test__ = False

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0], high=[1.0]),
    )


class _RecorderTestVariation(BuildTimeVariationBase):
    """Minimal build-time variation that just draws a sample when applied."""

    __test__ = False
    supported_sample_per_environment = (False, True)

    cfg: _RecorderTestVariationCfg

    def __init__(self, cfg: _RecorderTestVariationCfg | None = None, name: str = "recorder_test"):
        super().__init__(cfg=cfg if cfg is not None else _RecorderTestVariationCfg(), name=name)

    def _realize_at_build_time(self, context=None) -> None:
        if self.cfg.sample_per_environment:
            assert context is not None
            self.sampler.sample(num_samples=context.num_envs, env_ids=torch.arange(context.num_envs))
        else:
            self.sampler.sample(num_samples=1)


class _RecorderTestRuntimeVariation(RunTimeVariationBase):
    """Runtime variation whose sampler can be called without constructing a simulation."""

    def __init__(self):
        super().__init__(_RecorderTestVariationCfg(sample_per_environment=True), name="recorder_test")

    def build_event_cfg(self):
        raise NotImplementedError


def test_uniform_sampler_notifies_listeners():
    sampler = UniformSamplerCfg(low=[0.0], high=[1.0]).build()
    seen: list = []
    sampler.add_listener(lambda s, env_ids: seen.append((s, env_ids)))
    result = sampler.sample(num_samples=4, env_ids=[0, 1, 2, 3])
    assert len(seen) == 1
    sample, env_ids = seen[0]
    assert sample is result
    assert tuple(sample.shape) == (4, 1)
    # env_ids are forwarded so listeners can attribute each row to its env.
    assert env_ids == [0, 1, 2, 3]


def test_choice_sampler_notifies_listeners():
    sampler = ChoiceSampler()
    seen: list = []
    sampler.add_listener(lambda s, env_ids: seen.append((s, env_ids)))
    sampler.sample(num_samples=3, choices=["x", "y", "z"], env_ids=[4, 5, 6])
    assert len(seen) == 1
    sample, env_ids = seen[0]
    assert isinstance(sample, list) and len(sample) == 3
    assert env_ids == [4, 5, 6]


def test_variation_listener_survives_sampler_swap():
    """A listener added via the variation must re-bind to the sampler rebuilt by apply_cfg."""
    variation = _RecorderTestVariation()
    seen: list = []
    variation.add_sample_listener(lambda s, _env_ids: seen.append(s))

    # Swap in a new cfg (rebuilds the underlying sampler), then draw a sample.
    variation.apply_cfg(_RecorderTestVariationCfg(sampler_cfg=UniformSamplerCfg(low=[2.0], high=[2.0])))
    sample = variation.sampler.sample(num_samples=1)
    assert len(seen) == 1
    assert seen[0] is sample
    assert sample.item() == pytest.approx(2.0, abs=1e-6)


def test_recorder_records_build_time_sample_for_every_episode():
    """A build-time draw (env_ids=None) is recorded and surfaced for every env/episode."""
    variation = _RecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})

    record = recorder["asset.recorder_test"]
    assert record.sample_for_episode(0, 0) is None

    variation.configure_at_build_time()

    value = record.sample_for_episode(0, 0)
    # Tensor samples are stored detached on CPU.
    assert value.device.type == "cpu"
    # A build-time draw applies to every env and episode, not just (0, 0).
    assert record.sample_for_episode(7, 3).tolist() == value.tolist()


def test_variation_record_tracks_per_env_episode_values():
    """Each (env, episode) draw is stored and surfaced separately across envs and episodes."""
    record = VariationRecord(name="asset.var", cfg=_RecorderTestVariationCfg())

    # Runtime-style draw: rows map to the given env ids, all in episode 0.
    record.record_runtime_sample(torch.tensor([[1.0], [2.0]]), env_ids=[2, 5], episode_indices=[0, 0])
    assert record.sample_for_episode(2, 0).tolist() == [1.0]
    assert record.sample_for_episode(5, 0).tolist() == [2.0]
    # An env/episode not drawn for has no recorded value.
    assert record.sample_for_episode(0, 0) is None
    assert record.sample_for_episode(2, 1) is None

    # A draw for env 2 in the next episode lands under its own key.
    record.record_runtime_sample(torch.tensor([[4.0]]), env_ids=[2], episode_indices=[1])
    assert record.sample_for_episode(2, 1).tolist() == [4.0]
    assert record.sample_for_episode(2, 0).tolist() == [1.0]

    # Each (env, episode) is expected to be drawn for at most once.
    with pytest.raises(AssertionError):
        record.record_runtime_sample(torch.tensor([[3.0]]), env_ids=[2], episode_indices=[0])


@pytest.mark.parametrize("bind_before_sample", [False, True])
def test_recorder_keeps_per_environment_build_samples_across_episodes(bind_before_sample):
    from isaaclab_arena.recording.common_terms import record_variation_samples

    variation = _RecorderTestVariation(_RecorderTestVariationCfg(enabled=True, sample_per_environment=True))
    variation.set_replay_sampler(lambda _count, _env_ids: [[2.0], [5.0]])
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    env = _FakeEnv(recorder, episode_index=9)
    if bind_before_sample:
        recorder.bind_env(env)

    variation.configure_at_build_time(VariationBuildContext(num_envs=2))
    record = recorder["asset.recorder_test"]
    for episode_index in (0, 9, 25):
        assert record.sample_for_episode(0, episode_index).tolist() == [2.0]
        assert record.sample_for_episode(1, episode_index).tolist() == [5.0]
        assert record.sample_for_episode(2, episode_index) is None
    recorder.bind_env(env)
    assert record.sample_for_episode(0, 10).tolist() == [2.0]
    assert record_variation_samples(env, env_id=1) == {"variations": {"asset.recorder_test": [5.0]}}


@pytest.mark.parametrize("sample_per_environment", [False, True])
def test_build_time_recorder_requires_ids_matching_sampling_scope(sample_per_environment):
    variation = _RecorderTestVariation(
        _RecorderTestVariationCfg(enabled=True, sample_per_environment=sample_per_environment)
    )
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    incorrect_env_ids = None if sample_per_environment else torch.tensor([0])
    with pytest.raises(AssertionError, match="must agree with sample_per_environment"):
        variation.sampler.sample(num_samples=1, env_ids=incorrect_env_ids)
    assert recorder["asset.recorder_test"].sample_for_episode(0, 0) is None


@pytest.mark.parametrize("sample_per_environment", [False, True])
def test_build_time_record_rejects_duplicate_and_mixed_draws(sample_per_environment):
    cfg = _RecorderTestVariationCfg(sample_per_environment=sample_per_environment)
    record = VariationRecord(name="asset.var", cfg=cfg)
    env_ids = [2] if sample_per_environment else None
    record.record_buildtime_sample(["first"], env_ids=env_ids)

    with pytest.raises(AssertionError, match="already recorded"):
        record.record_buildtime_sample(["replacement"], env_ids=env_ids)
    cfg.sample_per_environment = not sample_per_environment
    other_env_ids = None if sample_per_environment else [5]
    with pytest.raises(AssertionError, match="already recorded"):
        record.record_buildtime_sample(["replacement"], env_ids=other_env_ids)
    assert record.sample_for_episode(2, 3) == "first"


@pytest.mark.parametrize("env_ids", [[2, 2], [2], [], [2, -1]])
def test_per_environment_build_record_rejects_invalid_rows_without_partial_recording(env_ids):
    record = VariationRecord(name="asset.var", cfg=_RecorderTestVariationCfg(sample_per_environment=True))
    with pytest.raises(AssertionError):
        record.record_buildtime_sample(["first", "second"], env_ids=env_ids)
    assert record.sample_for_episode(2, 0) is None


def test_per_environment_build_record_rejects_repeated_environment_without_partial_recording():
    record = VariationRecord(name="asset.var", cfg=_RecorderTestVariationCfg(sample_per_environment=True))
    record.record_buildtime_sample(["first"], env_ids=[2])
    with pytest.raises(AssertionError, match="already recorded"):
        record.record_buildtime_sample(["new", "replacement"], env_ids=[5, 2])
    assert record.sample_for_episode(2, 7) == "first"
    assert record.sample_for_episode(5, 7) is None
    record.record_buildtime_sample(["second"], env_ids=[5])
    assert record.sample_for_episode(5, 7) == "second"


def test_runtime_recorder_requires_environment_ids_and_binding():
    variation = _RecorderTestRuntimeVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    with pytest.raises(AssertionError, match="requires per-environment draws"):
        variation.sampler.sample(num_samples=1)
    with pytest.raises(AssertionError, match="needs bind_env"):
        variation.sampler.sample(num_samples=1, env_ids=torch.tensor([0]))
    assert recorder["asset.recorder_test"].sample_for_episode(0, 0) is None


class _FakeEnv:
    """Minimal stand-in exposing the attributes ``record_variation_samples`` reads."""

    def __init__(self, recorder: VariationRecorder, episode_index: int = 0) -> None:
        self.variation_recorder = recorder
        self._episode_index = episode_index

    def get_episode_index(self, env_id: int) -> int:  # noqa: ARG002
        return self._episode_index


def test_record_variation_samples_emits_the_per_episode_draw():
    """The episode term emits the draw made for the finishing episode, one value per variation."""
    from isaaclab_arena.recording.common_terms import record_variation_samples

    variation = _RecorderTestRuntimeVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    env = _FakeEnv(recorder, episode_index=0)
    recorder.bind_env(env)

    # One run-time draw for env 0 during episode 0.
    variation.sampler.sample(num_samples=1, env_ids=torch.tensor([0]))

    fields = record_variation_samples(env, env_id=0)
    sample = fields["variations"]["asset.recorder_test"]
    # The single (1,)-shaped draw is recorded as a flat list, not a list of draws.
    assert isinstance(sample, list) and len(sample) == 1

    # A different env in the same episode has nothing recorded, so no variations field is emitted.
    assert record_variation_samples(env, env_id=1) == {}


def test_recorder_skips_disabled_variations():
    variation = _RecorderTestVariation()  # disabled by default
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    assert "asset.recorder_test" not in recorder
    assert recorder.records == {}


def test_recorder_duplicate_asset_name_asserts():
    variation = _RecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})
    with pytest.raises(AssertionError):
        recorder.attach({"asset": [variation]})


def _test_hdr_variation_recorder_captures_chosen_hdr_name(simulation_app):
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.hdr_image_variation import HDRImageVariationCfg

    pool = ["home_office_robolab", "empty_warehouse_robolab", "billiard_hall_robolab"]

    dome_light = AssetRegistry().get_asset_by_name("light")()
    assert isinstance(dome_light, DomeLight)
    variation = dome_light.get_variation("hdr_image")
    variation.apply_cfg(HDRImageVariationCfg(enabled=True, hdr_names=pool))
    scene = Scene(assets=[dome_light])

    # Attach the recorder after the cfg is finalized but before the build, mirroring
    # the order ArenaEnvBuilder.compose_manager_cfg uses.
    recorder = VariationRecorder()
    recorder.attach(scene.get_asset_variations())

    record = recorder["light.hdr_image"]
    assert record.sample_for_episode(0, 0) is None

    variation.configure_at_build_time()

    # The HDR draw is build-time (env_ids=None), so it applies to every env/episode.
    value = record.sample_for_episode(0, 0)
    # The recorder must capture the chosen HDR *name*, not an index.
    assert value in pool
    return True


def test_hdr_variation_recorder_captures_chosen_hdr_name():
    assert run_function_with_persistent_simulation_app(
        _test_hdr_variation_recorder_captures_chosen_hdr_name,
        headless=HEADLESS,
    )
