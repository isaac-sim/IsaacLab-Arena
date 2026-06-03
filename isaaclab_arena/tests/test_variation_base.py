# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`VariationBase` subclassing and Cfg-driven configuration.

These tests stay in plain Python (no ``SimulationApp``) because they only
exercise the cfg plumbing and sampler wiring, not any Isaac Sim runtime.
"""

from dataclasses import field

import pytest
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg


def _noop_event(env, env_ids, asset_cfg):  # noqa: ARG001
    """No-op event term used to build ``EventTermCfg`` instances in these tests."""


@configclass
class _CustomVariationCfg(VariationBaseCfg):
    """Test-only cfg that adds tunables on top of :class:`VariationBaseCfg`."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0], high=[1.0, 1.0]),
    )

    asset_name: str = "test_asset"
    """Scene-entity name of the asset this variation targets."""

    scale: float = 1.0
    """Arbitrary tunable factor exposed through the cfg."""


class _CustomVariation(RunTimeVariationBase):
    """Minimal run-time variation used to exercise ``VariationBase`` plumbing."""

    name = "test_custom_variation"

    cfg: _CustomVariationCfg

    def __init__(self, cfg: _CustomVariationCfg | None = None):
        super().__init__(cfg=cfg if cfg is not None else _CustomVariationCfg())

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return (
            f"{self.cfg.asset_name}_{self.name}",
            EventTermCfg(
                func=_noop_event,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(self.cfg.asset_name)},
            ),
        )


def test_default_cfg_populates_variation_state():
    variation = _CustomVariation()

    assert isinstance(variation, RunTimeVariationBase)
    assert isinstance(variation.cfg, _CustomVariationCfg)
    assert variation.cfg.asset_name == "test_asset"
    assert variation.cfg.scale == 1.0
    assert variation.enabled is False
    assert isinstance(variation.sampler, UniformSampler)
    assert tuple(variation.sampler.shape_per_sample) == (2,)


def test_custom_cfg_flows_through_to_variation():
    cfg = _CustomVariationCfg(
        enabled=True,
        asset_name="cube",
        scale=2.5,
        sampler_cfg=UniformSamplerCfg(low=[-1.0], high=[1.0]),
    )
    variation = _CustomVariation(cfg)

    assert variation.cfg is cfg
    assert variation.enabled is True
    assert variation.cfg.asset_name == "cube"
    assert variation.cfg.scale == 2.5
    assert tuple(variation.sampler.shape_per_sample) == (1,)


def test_enable_disable_toggles_cfg_flag():
    variation = _CustomVariation()
    assert not variation.enabled

    variation.enable()
    assert variation.enabled and variation.cfg.enabled

    variation.disable()
    assert not variation.enabled and not variation.cfg.enabled


def test_apply_cfg_replaces_cfg_and_rebuilds_sampler():
    variation = _CustomVariation(
        _CustomVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0])),
    )
    original_sampler = variation.sampler

    new_cfg = _CustomVariationCfg(
        enabled=True,
        scale=10.0,
        sampler_cfg=UniformSamplerCfg(low=[5.0, 5.0, 5.0], high=[10.0, 10.0, 10.0]),
    )
    variation.apply_cfg(new_cfg)

    assert variation.cfg is new_cfg
    assert variation.enabled
    assert variation.cfg.scale == 10.0
    assert variation.sampler is not original_sampler
    assert tuple(variation.sampler.shape_per_sample) == (3,)
    samples = variation.sampler.sample(num_samples=8)
    assert (samples >= 5.0).all() and (samples <= 10.0).all()


def test_apply_cfg_with_only_sampler_change_rebuilds_sampler_in_place():
    """A new cfg that differs only in ``sampler_cfg`` should swap the live sampler and leave other state intact."""
    initial_cfg = _CustomVariationCfg(
        enabled=True,
        asset_name="cube",
        scale=3.14,
        sampler_cfg=UniformSamplerCfg(low=[0.0], high=[1.0]),
    )
    variation = _CustomVariation(initial_cfg)
    original_sampler = variation.sampler

    new_cfg = _CustomVariationCfg(
        enabled=initial_cfg.enabled,
        asset_name=initial_cfg.asset_name,
        scale=initial_cfg.scale,
        sampler_cfg=UniformSamplerCfg(low=[-2.0, -2.0], high=[-1.0, -1.0]),
    )
    variation.apply_cfg(new_cfg)

    assert variation.cfg is new_cfg
    assert variation.enabled is True
    assert variation.cfg.asset_name == "cube"
    assert variation.cfg.scale == 3.14

    assert variation.sampler is not original_sampler
    assert isinstance(variation.sampler, UniformSampler)
    assert tuple(variation.sampler.shape_per_sample) == (2,)
    samples = variation.sampler.sample(num_samples=16)
    assert (samples >= -2.0).all() and (samples <= -1.0).all()


def test_build_event_cfg_uses_configured_asset_name():
    variation = _CustomVariation(_CustomVariationCfg(asset_name="cube"))

    event_name, event_cfg = variation.build_event_cfg()

    assert event_name == f"cube_{_CustomVariation.name}"
    assert event_cfg.func is _noop_event
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == "cube"


def test_add_variation_raises_error_if_variation_name_is_already_attached():
    asset = Asset(name="test_asset")
    variation = _CustomVariation()
    asset.add_variation(variation)
    with pytest.raises(AssertionError):
        asset.add_variation(variation)
