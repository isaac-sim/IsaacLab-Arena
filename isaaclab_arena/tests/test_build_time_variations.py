# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the variation samplers and the build-time HDR-image variation.

Sampler-only tests are plain Python. End-to-end variation tests run inside
:func:`~isaaclab_arena.tests.utils.subprocess.run_simulation_app_function`
because constructing assets via the registry pulls in ``isaaclab.sim`` and
the asset library.
"""

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.categorical_sampler import CategoricalSampler, CategoricalSamplerCfg
from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg


def test_categorical_sampler_draws_items_from_choices():
    sampler = CategoricalSampler()
    choices = ["a", "b", "c", "d", "e"]
    samples = sampler.sample(num_samples=100, choices=choices)
    assert isinstance(samples, list)
    assert len(samples) == 100
    assert all(s in choices for s in samples)


def test_categorical_sampler_returns_actual_item_types():
    """Items in the returned list are the same objects (not copies / indices) as in choices."""
    sampler = CategoricalSampler()
    choices = [object(), object(), object()]
    [drawn] = sampler.sample(num_samples=1, choices=choices)
    assert any(drawn is c for c in choices)


def test_categorical_sampler_cfg_builds_live_sampler():
    assert isinstance(CategoricalSamplerCfg().build(), CategoricalSampler)


def test_categorical_sampler_rejects_empty_choices():
    sampler = CategoricalSampler()
    with pytest.raises(AssertionError):
        sampler.sample(num_samples=1, choices=[])


def test_uniform_sampler_draws_within_bounds():
    sampler = UniformSampler(low=[-0.005, 0.0, 1.0], high=[0.005, 0.0, 2.0])
    samples = sampler.sample(num_samples=128)
    assert samples.shape == (128, 3)
    assert (samples[:, 0] >= -0.005).all() and (samples[:, 0] <= 0.005).all()
    assert (samples[:, 1] == 0.0).all()
    assert (samples[:, 2] >= 1.0).all() and (samples[:, 2] <= 2.0).all()


def test_uniform_sampler_cfg_builds_live_sampler():
    sampler = UniformSamplerCfg(low=[0.0], high=[1.0]).build()
    assert isinstance(sampler, UniformSampler)
    assert tuple(sampler.shape_per_sample) == (1,)


def test_apply_cfg_keeps_sampler_cfg_and_live_sampler_in_sync():
    from isaaclab_arena.variations.camera_decalibration_variation import (
        CameraDecalibrationVariation,
        CameraDecalibrationVariationCfg,
    )

    variation = CameraDecalibrationVariation(
        "wrist_cam",
        cfg=CameraDecalibrationVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0])),
    )
    variation.apply_cfg(CameraDecalibrationVariationCfg(sampler_cfg=UniformSamplerCfg(low=[10.0], high=[20.0])))
    assert variation.cfg.sampler_cfg.low == [10.0] and variation.cfg.sampler_cfg.high == [20.0]
    samples = variation.sampler.sample(num_samples=16)
    assert (samples >= 10.0).all() and (samples <= 20.0).all()


def test_uniform_sampler_rejects_mismatched_bounds():
    with pytest.raises(AssertionError):
        UniformSampler(low=[0.0, 0.0], high=[1.0])
    with pytest.raises(AssertionError):
        UniformSampler(low=[1.0], high=[0.0])


def _test_hdr_variation_apply_mutates_dome_light(simulation_app):
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.hdr_image_variation import HDRImageVariation, HDRImageVariationCfg
    from isaaclab_arena.variations.variation_base import BuildTimeVariationBase

    asset_registry = AssetRegistry()
    light = asset_registry.get_asset_by_name("light")()
    assert isinstance(light, DomeLight)

    variation = light.get_variation("hdr_image")
    assert isinstance(variation, HDRImageVariation)
    assert isinstance(variation, BuildTimeVariationBase)
    assert not variation.enabled

    variation.apply_cfg(HDRImageVariationCfg(enabled=True, hdr_names=["home_office_robolab"]))
    variation.apply()

    texture_file = light.spawner_cfg.texture_file
    assert texture_file is not None and texture_file.endswith(
        "home_office.exr"
    ), f"Expected DomeLight.spawner_cfg.texture_file to end with 'home_office.exr', got {texture_file!r}."
    # The asset cfg consumed by Isaac Lab's scene builder must reflect the
    # newly-bound spawner cfg, since add_hdr re-initialises object_cfg.
    assert light.object_cfg.spawn.texture_file == texture_file
    return True


def test_hdr_variation_apply_mutates_dome_light():
    assert run_simulation_app_function(_test_hdr_variation_apply_mutates_dome_light)


def _test_hdr_variation_unknown_name_asserts(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.hdr_image_variation import HDRImageVariationCfg

    light = AssetRegistry().get_asset_by_name("light")()

    variation = light.get_variation("hdr_image")
    variation.apply_cfg(HDRImageVariationCfg(enabled=True, hdr_names=["does_not_exist_hdr"]))

    try:
        variation.apply()
    except AssertionError as err:
        assert "does_not_exist_hdr" in str(err)
        return True
    return False


def test_hdr_variation_unknown_name_asserts():
    assert run_simulation_app_function(_test_hdr_variation_unknown_name_asserts)
