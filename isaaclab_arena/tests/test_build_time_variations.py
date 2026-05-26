# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the build-time variation flavor and the HDR-image variation.

Sampler-only tests are plain Python. End-to-end variation tests run inside
:func:`~isaaclab_arena.tests.utils.subprocess.run_simulation_app_function`
because constructing assets via the registry pulls in ``isaaclab.sim`` and
the asset library.
"""

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.categorical_sampler import CategoricalSampler, CategoricalSamplerCfg


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


def test_categorical_sampler_notifies_listeners():
    sampler = CategoricalSampler()
    seen: list = []
    sampler.add_listener(lambda s: seen.append(s))
    sampler.sample(num_samples=3, choices=["x", "y", "z"])
    assert len(seen) == 1
    assert isinstance(seen[0], list) and len(seen[0]) == 3


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
    assert texture_file is not None and texture_file.endswith("home_office.exr"), (
        f"Expected DomeLight.spawner_cfg.texture_file to end with 'home_office.exr', got {texture_file!r}."
    )
    # The asset cfg consumed by Isaac Lab's scene builder must reflect the
    # newly-bound spawner cfg, since add_hdr re-initialises object_cfg.
    assert light.object_cfg.spawn.texture_file == texture_file
    return True


def test_hdr_variation_apply_mutates_dome_light():
    assert run_simulation_app_function(_test_hdr_variation_apply_mutates_dome_light)


def _test_hdr_variation_hydra_override_round_trip(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations import variations_hydra

    asset_registry = AssetRegistry()
    light = asset_registry.get_asset_by_name("light")()
    scene = Scene(assets=[light])

    variations_hydra.apply_overrides(
        scene.get_asset_variations(),
        [
            "light.hdr_image.enabled=true",
            "light.hdr_image.hdr_names=[empty_warehouse_robolab]",
        ],
    )

    variation = light.get_variation("hdr_image")
    assert variation.enabled
    assert list(variation.cfg.hdr_names) == ["empty_warehouse_robolab"]

    variation.apply()
    assert light.spawner_cfg.texture_file is not None
    assert light.spawner_cfg.texture_file.endswith("empty_warehouse.hdr")
    return True


def test_hdr_variation_hydra_override_round_trip():
    assert run_simulation_app_function(_test_hdr_variation_hydra_override_round_trip)


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


def _test_hdr_variation_recorder_captures_chosen_hdr_name(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations import variations_hydra
    from isaaclab_arena.variations.variations_recorder import VariationRecorder

    light = AssetRegistry().get_asset_by_name("light")()
    scene = Scene(assets=[light])

    pool = ["home_office_robolab", "empty_warehouse_robolab", "billiard_hall_robolab"]
    variations_hydra.apply_overrides(
        scene.get_asset_variations(),
        [
            "light.hdr_image.enabled=true",
            f"light.hdr_image.hdr_names=[{','.join(pool)}]",
        ],
    )

    # Attach the recorder *after* Hydra overrides but *before* apply(), which
    # mirrors the order ArenaEnvBuilder.compose_manager_cfg uses.
    recorder = VariationRecorder()
    recorder.attach_to_scene(scene)

    record = recorder["light.hdr_image"]
    assert len(record.samples) == 0

    light.get_variation("hdr_image").apply()

    assert len(record.samples) == 1
    sample = record.samples[0]
    # The recorder must capture the chosen HDR *name*, not an index.
    assert isinstance(sample, list) and len(sample) == 1
    assert sample[0] in pool
    return True


def test_hdr_variation_recorder_captures_chosen_hdr_name():
    assert run_simulation_app_function(_test_hdr_variation_recorder_captures_chosen_hdr_name)
