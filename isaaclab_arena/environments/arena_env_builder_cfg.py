# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

from dataclasses import dataclass
from enum import Enum

from isaaclab_arena.utils.physics_backend import PhysicsBackend


class CameraResolutionMode(str, Enum):
    """Which component decides the image size the embodiment cameras render."""

    EMBODIMENT_DEFAULT = "embodiment_default"
    """Keep every camera at the dimensions the embodiment's camera rig declares."""

    POLICY_DEFAULT = "policy_default"
    """Shrink every camera, keeping its aspect ratio, to fit inside the image size the policy declares."""

    def __str__(self) -> str:
        return self.value


# TODO(cvolk, 2026-07-06): [typed-config-migration] Replace this flat legacy-CLI-shaped configuration with
# nested scene, placement, and physics configs once the typed run configuration
# owns configuration composition.
@dataclass
class ArenaEnvBuilderCfg:
    """Configure how Arena builds an Isaac Lab environment."""

    num_envs: int = 1
    env_spacing: float = 30.0
    seed: int = 42
    solve_relations: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    disable_fabric: bool = False
    mimic: bool = False
    presets: PhysicsBackend | None = None
    device: str = "cuda:0"
    language_instruction: str | None = None
    record_trajectories: bool = False
    """Whether to extend the recorder manager with per-step trajectory terms."""
    recorder_dataset_export_dir_path: str | None = None
    """If set, overrides the recorder manager's dataset export directory."""
    recorder_dataset_filename: str | None = None
    """If set, overrides the recorder manager's dataset filename."""

    recorded_variation_samples_path: str | None = None
    """Episode-result JSONL containing variation samples to replay; relative to the working directory."""

    camera_resolution_mode: CameraResolutionMode = CameraResolutionMode.EMBODIMENT_DEFAULT
    """Which component decides the image size the embodiment cameras render.

    ``embodiment_default`` keeps the dimensions the camera rig declares; ``policy_default`` shrinks each
    camera, keeping its aspect ratio, to fit inside the image size the policy declares through
    ``PolicyBase.default_image_size``.
    """

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
        if self.presets is not None:
            self.presets = PhysicsBackend(self.presets)
        self.camera_resolution_mode = CameraResolutionMode(self.camera_resolution_mode)
