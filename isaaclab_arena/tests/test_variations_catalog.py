# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.test_build_time_variations import TEST_ASSET_NAME, TestBuildTimeVariation
from isaaclab_arena.variations.variations_catalog import get_variations_catalogue_as_string


class _MockHost:
    """Minimal variation host for catalog tests (no Scene / USD imports)."""

    name = TEST_ASSET_NAME


def _variations_mapping(*, enabled: bool = False) -> dict[str, list[TestBuildTimeVariation]]:
    host = _MockHost()
    variation = TestBuildTimeVariation(host)
    if enabled:
        variation.enable()
    return {TEST_ASSET_NAME: [variation]}


def test_catalog_lists_variation_paths():
    text = get_variations_catalogue_as_string(_variations_mapping())

    assert "build-time" in text
    assert f"{TEST_ASSET_NAME}.test_build_time.enabled=true" in text
    assert "sampler_cfg" in text
    assert "TestBuildTimeVariation" in text


def test_catalog_reflects_hydra_overrides():
    text = get_variations_catalogue_as_string(
        _variations_mapping(),
        hydra_overrides=[f"{TEST_ASSET_NAME}.test_build_time.enabled=true"],
    )
    assert "(default: True)" in text


def test_empty_variations_message():
    assert get_variations_catalogue_as_string({}) == "No variations attached to this environment.\n"
