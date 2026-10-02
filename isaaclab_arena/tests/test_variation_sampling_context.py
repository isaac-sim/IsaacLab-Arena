# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise independent reset draws and recorded-value replay without SimulationApp."""

import json
import torch
from copy import deepcopy
from dataclasses import field
from types import SimpleNamespace

import pytest
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.bernoulli_sampler import BernoulliSampler
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler, ContinuousSamplerCfg
from isaaclab_arena.variations.sampling_context import VariationReplay, VariationSamplingContext
from isaaclab_arena.variations.uniform_sampler import UniformSampler


def _context(seed=19, replay=None, episodes=None):
    context = VariationSamplingContext(seed=seed, replay=replay)
    episodes = episodes or {0: 0, 1: 2, 2: 3}
    context.bind_env(SimpleNamespace(get_episode_index=lambda env_id: episodes[env_id]))
    return context


def _sampler(family):
    if family == "uniform":
        return UniformSampler([-2, 1], [3, 4]), {}
    if family == "choice":
        return ChoiceSampler(), {"choices": ["first", "second", "third"]}
    return BernoulliSampler(0.4), {}


def _plain(value):
    return value.tolist() if isinstance(value, torch.Tensor) else value


@pytest.mark.parametrize("family", ["uniform", "choice", "bernoulli"])
def test_keyed_samples_ignore_global_rng_other_variations_and_reset_order(family):
    sampler, kwargs = _sampler(family)
    context = _context()
    sampler.bind_sampling_context(context, "asset.test")
    before = torch.random.get_rng_state().clone()
    expected = _plain(sampler.sample(3, env_ids=torch.tensor([0, 1, 2]), **kwargs))
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(37)
    unrelated = UniformSampler([0], [1])
    unrelated.bind_sampling_context(context, "other.test")
    unrelated.sample(2, env_ids=torch.tensor([2, 0]))
    actual = _plain(sampler.sample(2, env_ids=torch.tensor([2, 0]), **kwargs))
    assert actual == [expected[2], expected[0]]
    assert _plain(sampler.sample(1, env_ids=[1], **kwargs)) == [expected[1]]


def test_path_episode_and_seed_change_independent_draws():
    episodes = {0: 0}
    context = _context(episodes=episodes)
    sampler = UniformSampler([0] * 6, [1] * 6)
    sampler.bind_sampling_context(context, "asset.test")
    first = sampler.sample(1, [0])
    episodes[0] = 1
    assert not torch.equal(first, sampler.sample(1, [0]))
    episodes[0] = 0
    sampler.bind_sampling_context(context, "asset.other")
    assert not torch.equal(first, sampler.sample(1, [0]))
    sampler.bind_sampling_context(_context(seed=20), "asset.test")
    assert not torch.equal(first, sampler.sample(1, [0]))


@pytest.mark.parametrize("family", ["uniform", "choice", "bernoulli"])
def test_recorded_values_replay_once_through_sampler_listeners(family, tmp_path):
    source, kwargs = _sampler(family)
    source.bind_sampling_context(_context(), "asset.test")
    expected = _plain(source.sample(2, env_ids=[0, 1], **kwargs))
    rows = [
        {"env_id": 0, "episode_in_env": 0, "variations": {"asset.test": expected[0]}},
        {"env_id": 1, "episode_in_env": 2, "variations": {"asset.test": expected[1]}},
    ]
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    replay = VariationReplay.from_jsonl(path)
    target, kwargs = _sampler(family)
    target.bind_sampling_context(_context(seed=None, replay=replay), "asset.test")
    seen = []
    target.add_listener(lambda sample, ids: seen.append((sample, ids)))
    before = torch.random.get_rng_state().clone()
    actual = target.sample(2, env_ids=[1, 0], **kwargs)
    assert _plain(actual) == list(reversed(expected))
    assert len(seen) == 1 and seen[0][0] is actual
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("value", [[True, 2], [1], [[1, 2]], [float("nan"), 2], [8, 2], "bad"])
def test_invalid_numeric_replay_never_notifies(value):
    replay = VariationReplay([{"env_id": 0, "episode_in_env": 0, "variations": {"asset.test": value}}])
    sampler = UniformSampler([-2, 1], [3, 4])
    sampler.bind_sampling_context(_context(replay=replay), "asset.test")
    sampler.add_listener(lambda *_: pytest.fail("Invalid replay must not reach listeners"))
    with pytest.raises((AssertionError, ValueError)):
        sampler.sample(1, [0])


@pytest.mark.parametrize(
    "family,value",
    [("choice", "unknown"), ("choice", True), ("bernoulli", 1), ("bernoulli", [True])],
)
def test_replay_enforces_categorical_types(family, value):
    replay = VariationReplay([{"env_id": 0, "episode_in_env": 0, "variations": {"asset.test": value}}])
    sampler, kwargs = _sampler(family)
    sampler.bind_sampling_context(_context(replay=replay), "asset.test")
    with pytest.raises(AssertionError):
        sampler.sample(1, env_ids=[0], **kwargs)


def test_missing_values_do_not_fall_back_to_random_sampling():
    replay = VariationReplay([{"env_id": 0, "episode_in_env": 0, "variations": {}}])
    sampler = UniformSampler([0], [1])
    sampler.bind_sampling_context(_context(seed=4, replay=replay), "asset.test")
    with pytest.raises(AssertionError, match="Missing replay value"):
        sampler.sample(1, [0])
    with pytest.raises(AssertionError, match="Missing variation replay episode"):
        sampler.sample(1, [1])


def test_build_time_replay_requires_consistent_rows():
    rows = [
        {"env_id": 0, "episode_in_env": 0, "variations": {"asset.test": [0.2]}},
        {"env_id": 1, "episode_in_env": 0, "variations": {"asset.test": [0.2]}},
    ]
    sampler = UniformSampler([0], [1])
    sampler.bind_sampling_context(_context(replay=VariationReplay(rows)), "asset.test")
    assert sampler.sample(1).item() == pytest.approx(0.2)
    rows[1]["variations"]["asset.test"] = [0.4]
    sampler.bind_sampling_context(_context(replay=VariationReplay(rows)), "asset.test")
    with pytest.raises(AssertionError, match="disagree"):
        sampler.sample(1)


def test_event_config_deepcopy_shares_lifecycle_binding():
    context = VariationSamplingContext(seed=8)
    sampler = UniformSampler([0], [1])
    sampler.bind_sampling_context(context, "asset.test")
    copied = deepcopy(sampler)
    context.bind_env(SimpleNamespace(get_episode_index=lambda _env_id: 3))
    assert torch.equal(sampler.sample(1, [0]), copied.sample(1, [0]))


def test_bound_context_does_not_recurse_through_environment_config():
    from isaaclab.managers import EventTermCfg

    context = VariationSamplingContext(seed=8)
    sampler = UniformSampler([0], [1])
    sampler.bind_sampling_context(context, "asset.test")
    cfg = EventTermCfg(func=lambda *_: None, mode="reset", params={"sampler": sampler})
    context.bind_env(SimpleNamespace(cfg=cfg, get_episode_index=lambda _: 0))
    cfg.validate()


@pytest.mark.parametrize("ids,count", [([0, 0], 2), ([0], 2), ([-1], 1), ([True], 1), (None, 2)])
def test_invalid_sample_keys_are_rejected(ids, count):
    sampler = UniformSampler([0], [1])
    sampler.bind_sampling_context(_context(), "asset.test")
    with pytest.raises(AssertionError):
        sampler.sample(count, ids)


def test_legacy_sampler_still_follows_torch_seed():
    sampler = UniformSampler([0], [1])
    torch.manual_seed(11)
    first = sampler.sample(4)
    torch.manual_seed(11)
    assert torch.equal(first, sampler.sample(4))


def test_replay_rejects_duplicate_episode_keys():
    row = {"env_id": 0, "episode_in_env": 0, "variations": {"asset.test": [0.2]}}
    with pytest.raises(AssertionError, match="Duplicate"):
        VariationReplay([row, row])


def test_variation_binding_survives_config_swap_and_replays_application():
    from isaaclab_arena.variations.light_intensity_variation import LightIntensityVariation, LightIntensityVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
    from isaaclab_arena.variations.variation_recorder import VariationRecorder

    applied = []
    light = SimpleNamespace(set_intensity=applied.append)
    variation = LightIntensityVariation(light)
    variation.bind_sampling_context(VariationSamplingContext(seed=81), "light.intensity")
    cfg = LightIntensityVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[20], high=[30]))
    variation.apply_cfg(cfg)
    recorder = VariationRecorder()
    recorder.attach({"light": [variation]})
    variation.configure_at_build_time()
    recorded = recorder["light.intensity"].sample_for_episode(0, 0).tolist()
    assert applied == recorded
    replay = VariationReplay([{
        "env_id": 0,
        "episode_in_env": 0,
        "variations": {"light.intensity": recorded},
    }])
    target = LightIntensityVariation(light, cfg)
    target.bind_sampling_context(VariationSamplingContext(replay=replay), "light.intensity")
    target.configure_at_build_time()
    assert applied == recorded * 2


def test_runtime_replay_is_attributed_to_one_finished_episode():
    from isaaclab_arena.variations.light_intensity_variation import LightIntensityVariation, LightIntensityVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
    from isaaclab_arena.variations.variation_recorder import VariationRecorder

    cfg = LightIntensityVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[20], high=[30]))
    variation = LightIntensityVariation(None, cfg)
    replay = VariationReplay([{"env_id": 1, "episode_in_env": 2, "variations": {"light.intensity": [24.0]}}])
    context = _context(replay=replay)
    variation.bind_sampling_context(context, "light.intensity")
    recorder = VariationRecorder()
    recorder.attach({"light": [variation]})
    recorder.bind_env(SimpleNamespace(get_episode_index=lambda _: 2))
    variation.sampler.sample(1, [1])
    assert recorder["light.intensity"].sample_for_episode(1, 2).tolist() == [24.0]
    assert recorder["light.intensity"].sample_for_episode(0, 2) is None


@pytest.mark.parametrize(
    "kind,low,high",
    [
        ("mass", [0], [1]),
        ("intrinsics", [-1, 0], [0, 0]),
        ("extrinsics", [0, 0], [1, 1]),
        ("intensity", [-1], [100]),
        ("temperature", [0], [1000]),
        ("color", [0, 0, 0], [1, 2, 1]),
    ],
)
def test_physical_domains_reject_invalid_configs_before_application(kind, low, high):
    from isaaclab_arena.variations.camera_extrinsics_variation import (
        CameraExtrinsicsVariation,
        CameraExtrinsicsVariationCfg,
    )
    from isaaclab_arena.variations.camera_intrinsics_variation import (
        CameraIntrinsicsVariation,
        CameraIntrinsicsVariationCfg,
    )
    from isaaclab_arena.variations.light_color_temperature_variation import (
        LightColorTemperatureVariation,
        LightColorTemperatureVariationCfg,
    )
    from isaaclab_arena.variations.light_color_variation import LightColorVariation, LightColorVariationCfg
    from isaaclab_arena.variations.light_intensity_variation import LightIntensityVariation, LightIntensityVariationCfg
    from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation, ObjectMassVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    sampler_cfg = UniformSamplerCfg(low=low, high=high)
    constructors = {
        "mass": lambda: ObjectMassVariation("part", ObjectMassVariationCfg(sampler_cfg=sampler_cfg)),
        "intrinsics": lambda: CameraIntrinsicsVariation(
            "camera", None, CameraIntrinsicsVariationCfg(sampler_cfg=sampler_cfg)
        ),
        "extrinsics": lambda: CameraExtrinsicsVariation(
            "camera", CameraExtrinsicsVariationCfg(sampler_cfg=sampler_cfg)
        ),
        "intensity": lambda: LightIntensityVariation(None, LightIntensityVariationCfg(sampler_cfg=sampler_cfg)),
        "temperature": lambda: LightColorTemperatureVariation(
            None, LightColorTemperatureVariationCfg(sampler_cfg=sampler_cfg)
        ),
        "color": lambda: LightColorVariation(None, LightColorVariationCfg(sampler_cfg=sampler_cfg)),
    }
    with pytest.raises(AssertionError):
        constructors[kind]().configure_at_build_time()


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_uniform_domain_is_rejected(bound):
    with pytest.raises(AssertionError, match="finite"):
        UniformSampler([bound], [bound])


def test_sample_trace_replays_final_autoreset_without_an_episode_result(tmp_path):
    from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation
    from isaaclab_arena.variations.light_intensity_variation import LightIntensityVariation, LightIntensityVariationCfg
    from isaaclab_arena.variations.variation_recorder import VariationRecorder

    episodes = {0: 0}
    env = SimpleNamespace(get_episode_index=lambda env_id: episodes[env_id])
    context = VariationSamplingContext(seed=49)
    context.bind_env(env)
    camera = CameraExtrinsicsVariation("camera")
    camera.enable()
    camera.bind_sampling_context(context, "robot.camera")
    light = LightIntensityVariation(
        SimpleNamespace(set_intensity=lambda _: None), LightIntensityVariationCfg(enabled=True)
    )
    light.bind_sampling_context(context, "light.intensity")
    recorder = VariationRecorder()
    recorder.attach({"robot": [camera], "light": [light]})
    # The recorder keys follow variation names, as they do in the builder.
    camera.bind_sampling_context(context, "robot.camera_extrinsics_camera")
    recorder.bind_env(env)
    light.configure_at_build_time()
    initial = camera.sampler.sample(1, [0])
    episodes[0] = 1
    final_autoreset = camera.sampler.sample(1, [0])
    path = tmp_path / "variation_samples_rebuild0.jsonl"
    recorder.write_samples_jsonl(path)
    replay = VariationReplay.from_jsonl(path)
    target = CameraExtrinsicsVariation("camera")
    replay_context = VariationSamplingContext(replay=replay)
    replay_context.bind_env(env)
    target.bind_sampling_context(replay_context, "robot.camera_extrinsics_camera")
    assert torch.equal(target.sampler.sample(1, [0]), final_autoreset)
    episodes[0] = 0
    assert torch.equal(target.sampler.sample(1, [0]), initial)
    assert replay.value("light.intensity", None) == recorder["light.intensity"].sample_for_episode(0, 0).tolist()
    episodes[0] = 2
    with pytest.raises(AssertionError, match="Missing variation replay episode"):
        target.sampler.sample(1, [0])


@pytest.mark.parametrize(
    "rows",
    [
        [{"variation": "a.b", "scope": "invalid", "value": 1}],
        [{"variation": "a.b", "scope": "build"}],
        [{"variation": "a.b", "scope": "runtime", "env_id": True, "episode_in_env": 0, "value": 1}],
        [{"variation": "a.b", "scope": "build", "value": 1}] * 2,
        [{"variation": "a.b", "scope": "runtime", "env_id": 0, "episode_in_env": 0, "value": 1}] * 2,
        [
            {"variation": "a.b", "scope": "build", "value": 1},
            {"variation": "a.b", "scope": "runtime", "env_id": 0, "episode_in_env": 0, "value": 1},
        ],
    ],
)
def test_invalid_sample_trace_rejected(rows, tmp_path):
    path = tmp_path / "samples.jsonl"
    header = {"schema": "arena.variation_samples", "version": 1}
    path.write_text("\n".join(json.dumps(row) for row in [header, *rows]))
    with pytest.raises(AssertionError):
        VariationReplay.from_jsonl(path)


class _LegacyConstantSampler(ContinuousSampler):
    """Legacy custom sampler implementing only the original continuous-sampler hooks."""

    def __init__(self, values):
        super().__init__()
        self.values = values
        self.draws = 0

    @property
    def shape_per_sample(self):
        return torch.Size((len(self.values),))

    def _sample(self, num_samples):
        self.draws += 1
        return torch.tensor(self.values, dtype=torch.float32).expand(num_samples, -1).clone()


@configclass
class _LegacyConstantSamplerCfg(ContinuousSamplerCfg):
    values: list[float] = field(default_factory=lambda: [0.2])

    def build(self):
        return _LegacyConstantSampler(self.values)


def _custom_continuous_variation(kind, values):
    from importlib import import_module

    from isaaclab_arena.utils.configclass import make_configclass

    definitions = {
        "mass": ("object_mass_variation", "ObjectMassVariation", ("part",)),
        "intrinsics": ("camera_intrinsics_variation", "CameraIntrinsicsVariation", ("camera", None)),
        "extrinsics": ("camera_extrinsics_variation", "CameraExtrinsicsVariation", ("camera",)),
        "intensity": ("light_intensity_variation", "LightIntensityVariation", (None,)),
        "temperature": ("light_color_temperature_variation", "LightColorTemperatureVariation", (None,)),
        "color": ("light_color_variation", "LightColorVariation", (None,)),
        "direction": ("light_direction_variation", "LightDirectionVariation", (None,)),
    }
    module_name, class_name, args = definitions[kind]
    module = import_module(f"isaaclab_arena.variations.{module_name}")
    cfg_type = make_configclass(
        "Custom" + class_name + "Cfg",
        [("sampler_cfg", _LegacyConstantSamplerCfg, _LegacyConstantSamplerCfg(values=values))],
        bases=(getattr(module, class_name + "Cfg"),),
    )
    cfg = cfg_type()
    return getattr(module, class_name)(*args, cfg=cfg)


@pytest.mark.parametrize(
    "kind,values",
    [
        ("mass", [0.2]),
        ("intrinsics", [0.1, -0.05]),
        ("extrinsics", [0.001, 0, 0]),
        ("intensity", [600]),
        ("temperature", [5000]),
        ("color", [0.3, 0.5, 0.9]),
        ("direction", [0.3, 0.2]),
    ],
)
def test_legacy_custom_sampler_remains_usable_and_discoverable_without_preflight_draws(kind, values):
    from isaaclab_arena.variations.variations_printing import get_variations_catalogue_as_dict

    variation = _custom_continuous_variation(kind, values)
    variation.validate_cfg()
    catalogue = get_variations_catalogue_as_dict({"asset": [variation]})
    entry = catalogue["variations"][0]
    assert entry["fields"][f"asset.{variation.name}.sampler_cfg.values"]["value"] == values
    assert variation.sampler.draws == 0, "Preflight and discovery must not probe a custom distribution."
    seen = []
    variation.add_sample_listener(lambda sample, ids: seen.append((sample, ids)))
    sample = variation.sampler.sample(2, env_ids=[1, 0])
    torch.testing.assert_close(sample, torch.tensor([values, values], dtype=torch.float32))
    assert variation.sampler.draws == 1
    assert len(seen) == 1 and seen[0][0] is sample and seen[0][1] == [1, 0]


@pytest.mark.parametrize(
    "kind,values",
    [
        ("mass", [-0.1]),
        ("intrinsics", [-1, 0]),
        ("extrinsics", [float("inf"), 0, 0]),
        ("intensity", [-1]),
        ("temperature", [0]),
        ("color", [1.1, 0.2, 0.3]),
        ("direction", [float("nan"), 0]),
    ],
)
def test_custom_sampler_invalid_realized_domain_never_reaches_recorders(kind, values):
    variation = _custom_continuous_variation(kind, values)
    variation.validate_cfg()
    variation.add_sample_listener(lambda *_: pytest.fail("An invalid realized sample cannot be recorded."))
    with pytest.raises(AssertionError):
        variation.sampler.sample(1)


def test_custom_sampler_keyed_generation_requires_explicit_opt_in_but_replay_does_not():
    variation = _custom_continuous_variation("mass", [0.2])
    variation.validate_cfg()
    variation.bind_sampling_context(VariationSamplingContext(seed=8), "part.mass")
    with pytest.raises(NotImplementedError, match="contextual sampling"):
        variation.sampler.sample(1)
    replay = VariationReplay([], build_values={"part.mass": [0.3]})
    variation.bind_sampling_context(VariationSamplingContext(replay=replay), "part.mass")
    assert variation.sampler.sample(1).item() == pytest.approx(0.3)
    replay = VariationReplay([], build_values={"part.mass": [-0.3]})
    variation.bind_sampling_context(VariationSamplingContext(replay=replay), "part.mass")
    with pytest.raises(AssertionError, match="physical minimum"):
        variation.sampler.sample(1)
    assert variation.sampler.draws == 0
