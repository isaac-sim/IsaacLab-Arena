# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for offline placement recording."""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING

from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams


@dataclass
class PlacementRecordingCfg:
    """Source scene, reset sampling and offline recording settings."""

    env_spec: str = MISSING
    """Environment YAML path."""
    output: str = MISSING
    """Placement JSONL output path; must not exist."""
    num_envs: int = 1
    """Number of parallel simulation environments."""
    env_spacing: float = 30.0
    """Distance between parallel environment origins, in metres."""
    viewer_eye: tuple[float, float, float] | None = None
    """Optional viewer position in simulation-world metres; requires viewer_lookat."""
    viewer_lookat: tuple[float, float, float] | None = None
    """Optional viewer target in simulation-world metres; requires viewer_eye."""
    layouts_per_env: int = 5
    """Number of reset placements sampled per environment before physics filtering."""
    seed: int = 42
    """Seed for placement solving and reset randomization."""
    presets: str | None = None
    """Optional physics backend override: physx or newton."""
    render: bool = False
    """Render physics steps when a visualizer is enabled."""
    settle: PlacementRecordingParams = field(default_factory=PlacementRecordingParams)
    """Physics duration, configured validators and minimum accepted count."""


def load_recording_config(overrides: list[str]) -> PlacementRecordingCfg:
    """Load recording settings from Hydra override tokens.

    Args:
        overrides: Recording field overrides in Hydra key=value syntax.

    Returns:
        Typed recording settings with required fields resolved.
    """
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    ConfigStore.instance().store(name="placement_recording", node=PlacementRecordingCfg)
    with initialize(version_base=None, config_path=None):
        return OmegaConf.to_object(compose(config_name="placement_recording", overrides=overrides))
