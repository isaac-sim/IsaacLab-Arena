# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation discovery preserves effective configuration without drawing samples."""

import json

from isaaclab_arena.agentic_environment_generation.authoring_metadata import AuthoringMetadata, ParameterMetadata
from isaaclab_arena.variations.object_mass_variation import ObjectMassVariation
from isaaclab_arena.variations.variations_printing import get_variations_catalogue_as_dict


class _MassVariation(ObjectMassVariation):
    authoring_metadata = AuthoringMetadata(
        configuration={"sampler_cfg.low": ParameterMetadata(units="kg", minimum=1e-6)},
        reset_semantics="Samples mass for resetting environments only.",
    )


def test_variation_catalogue_exposes_effective_paths_types_units_and_task_restrictions(monkeypatch):
    variation = _MassVariation("part")

    def fail(*args, **kwargs):
        raise AssertionError("Variation discovery must not sample")

    monkeypatch.setattr(variation.sampler, "sample", fail)
    catalogue = get_variations_catalogue_as_dict(
        {"part": [variation]},
        hydra_overrides=["part.mass.enabled=true", "part.mass.sampler_cfg.low=[0.2]"],
        restrictions={"part.mass": "Measured mass is part of the task's latent condition."},
    )
    entry = catalogue["variations"][0]
    assert entry["path"] == "part.mass"
    assert entry["enable_path"] == "part.mass.enabled"
    assert entry["enabled"]
    assert entry["timing"] == "run-time"
    assert not entry["supported"]
    assert entry["restriction_reason"] == "Measured mass is part of the task's latent condition."
    field = entry["fields"]["part.mass.sampler_cfg.low"]
    assert field["type"] == "array"
    assert field["items"] == {"type": "number", "minimum": 1e-6}
    assert field["default"] == [0.05]
    assert field["value"] == [0.2]
    assert field["x-units"] == "kg"
    assert entry["effective_config"]["sampler_cfg"]["low"] == [0.2]
    assert not variation.enabled
    assert variation.cfg.sampler_cfg.low == [0.05]
    json.dumps(catalogue, allow_nan=False)


def test_empty_variation_catalogue_is_machine_readable():
    assert get_variations_catalogue_as_dict({}) == {"schema_version": 1, "variations": []}
