# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for compiling an Arena environment."""

import re
from dataclasses import dataclass

from isaaclab_arena.utils.physics_backend import PhysicsBackend

CAMERA_RESOLUTION_HD = "hd"
"""Keep every embodiment camera at the dimensions its rig declares."""

CAMERA_RESOLUTION_AUTO = "auto"
"""Fit every embodiment camera into the image size the policy consumes, preserving its aspect ratio."""

_EXPLICIT_CAMERA_RESOLUTION_PATTERN = re.compile(r"^(?P<height>\d+)x(?P<width>\d+)$")


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

    camera_resolution: str = CAMERA_RESOLUTION_HD
    """Image dimensions rendered by every embodiment camera.

    ``"hd"`` keeps the rig's declared dimensions, ``"auto"`` fits each camera into the image size the
    policy consumes (so the policy-side letterbox only pads), and ``"<height>x<width>"`` such as
    ``"126x224"`` sets exact dimensions.
    """

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"
        if self.presets is not None:
            self.presets = PhysicsBackend(self.presets)
        self.camera_resolution = str(self.camera_resolution)
        if self.camera_resolution not in (CAMERA_RESOLUTION_HD, CAMERA_RESOLUTION_AUTO):
            height, width = self.explicit_camera_resolution_hw()
            assert (
                height > 0 and width > 0
            ), f"camera_resolution dimensions must be greater than zero: '{self.camera_resolution}'"

    @property
    def camera_resolution_is_auto(self) -> bool:
        """Whether cameras are fitted to the policy's image size."""
        return self.camera_resolution == CAMERA_RESOLUTION_AUTO

    def explicit_camera_resolution_hw(self) -> tuple[int, int]:
        """Return the (height, width) named by an explicit ``"<height>x<width>"`` camera resolution."""
        match = _EXPLICIT_CAMERA_RESOLUTION_PATTERN.match(self.camera_resolution)
        assert match is not None, (
            f"camera_resolution must be '{CAMERA_RESOLUTION_HD}', '{CAMERA_RESOLUTION_AUTO}', or '<height>x<width>'"
            f" such as '126x224': '{self.camera_resolution}'"
        )
        return int(match["height"]), int(match["width"])
