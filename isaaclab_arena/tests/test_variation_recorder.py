# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for sample listeners and ``VariationRecorder``.

Listener and recorder mechanics are plain Python (no SimulationApp). The end-to-end test that
records a real HDR draw runs inside ``run_simulation_app_function`` because constructing a dome
light via the registry pulls in ``isaaclab.sim``.
"""

from dataclasses import field

import pytest
from isaaclab.utils import configclass

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg
from isaaclab_arena.variations.variation_recorder import VariationRecorder

HEADLESS = True


@configclass
class _RecorderTestVariationCfg(VariationBaseCfg):
    """Build-time variation cfg whose sampler draws a single scalar."""

    __test__ = False

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0], high=[1.0]),
    )


class _RecorderTestVariation(BuildTimeVariationBase):
    """Minimal build-time variation that just draws a sample when applied."""

    __test__ = False

    cfg: _RecorderTestVariationCfg

    def __init__(self, cfg: _RecorderTestVariationCfg | None = None, name: str = "recorder_test"):
        super().__init__(cfg=cfg if cfg is not None else _RecorderTestVariationCfg(), name=name)

    def apply(self) -> None:
        assert self.sampler is not None
        self.sampler.sample(num_samples=1)


def test_uniform_sampler_notifies_listeners():
    sampler = UniformSamplerCfg(low=[0.0], high=[1.0]).build()
    seen: list = []
    sampler.add_listener(lambda s: seen.append(s))
    result = sampler.sample(num_samples=4)
    assert len(seen) == 1
    assert seen[0] is result
    assert tuple(seen[0].shape) == (4, 1)


def test_choice_sampler_notifies_listeners():
    sampler = ChoiceSampler()
    seen: list = []
    sampler.add_listener(lambda s: seen.append(s))
    sampler.sample(num_samples=3, choices=["x", "y", "z"])
    assert len(seen) == 1
    assert isinstance(seen[0], list) and len(seen[0]) == 3


def test_variation_listener_survives_sampler_swap():
    """A listener added via the variation must re-bind to the sampler rebuilt by apply_cfg."""
    variation = _RecorderTestVariation()
    seen: list = []
    variation.add_sample_listener(lambda s: seen.append(s))

    # Swap in a new cfg (rebuilds the underlying sampler), then draw a sample.
    variation.apply_cfg(_RecorderTestVariationCfg(sampler_cfg=UniformSamplerCfg(low=[2.0], high=[2.0])))
    sample = variation.sampler.sample(num_samples=1)
    assert len(seen) == 1
    assert seen[0] is sample
    assert sample.item() == pytest.approx(2.0, abs=1e-6)


def test_recorder_records_samples_from_attached_variation():
    variation = _RecorderTestVariation()
    variation.enable()
    recorder = VariationRecorder()
    recorder.attach({"asset": [variation]})

    record = recorder["asset.recorder_test"]
    assert len(record.samples) == 0

    variation.apply()
    variation.apply()

    assert len(record.samples) == 2
    # Tensor samples are stored detached on CPU.
    assert record.samples[0].device.type == "cpu"


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

    # Attach the recorder after the cfg is finalized but before apply(), mirroring
    # the order ArenaEnvBuilder.compose_manager_cfg uses.
    recorder = VariationRecorder()
    recorder.attach(scene.get_asset_variations())

    record = recorder["light.hdr_image"]
    assert len(record.samples) == 0

    variation.apply()

    assert len(record.samples) == 1
    sample = record.samples[0]
    # The recorder must capture the chosen HDR *name*, not an index.
    assert isinstance(sample, list) and len(sample) == 1
    assert sample[0] in pool
    return True


def test_hdr_variation_recorder_captures_chosen_hdr_name():
    assert run_simulation_app_function(
        _test_hdr_variation_recorder_captures_chosen_hdr_name,
        headless=HEADLESS,
    )
