# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

from isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter import (
    GR00T_EMBODIMENT_ADAPTER_CLASS_PATHS,
    import_gr00t_embodiment_adapter_class,
)


@dataclass
class Gr00tClosedloopPolicyCfg:
    """Configure GR00T closed-loop policy translation and inference."""

    language_instruction: str = field(
        default="", metadata={"description": "Instruction given to the policy in natural language."}
    )
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    denoising_steps: int = field(
        default=4, metadata={"description": "Number of denoising steps used in the policy inference."}
    )
    modality_config_path: str = field(
        default=None, metadata={"description": "Path to the modality configuration file."}
    )
    original_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3),
        metadata={"description": "Target size for images after resizing and padding as (height, width, channels)."},
    )
    gr00t_embodiment_adapter: str | None = field(
        default=None,
        metadata={
            "description": (
                "Gr00tEmbodimentAdapter that translates Arena observations and actions for GR00T, and whose"
                " embodiment tag selects the modality config: a built-in name (droid, gr1_arms, g1_wbc) or a"
                " dotted class path."
            )
        },
    )
    # Default to GPU policy and CPU physics simulation
    policy_device: str = field(
        default="cuda", metadata={"description": "Device to run the policy model on (e.g., 'cuda' or 'cpu')."}
    )
    video_backend: str = field(default="decord", metadata={"description": "Video backend to use for evaluation."})
    pov_cam_name_sim: list[str] = field(
        default_factory=lambda: ["robot_head_cam_rgb"],
        metadata={"description": "Names of the POV cameras of the robot in simulation."},
    )
    # Closed loop specific parameters
    action_chunk_length: int = field(
        default=16,
        metadata={
            "description": "Number of actions to execute per inference rollout (can be less than action_horizon)."
        },
    )
    seed: int = field(default=10, metadata={"description": "Random seed for reproducibility."})

    def __post_init__(self):
        assert (
            self.action_chunk_length <= self.action_horizon
        ), "action_chunk_length must be less than or equal to action_horizon"
        if self.modality_config_path:
            assert Path(
                self.modality_config_path
            ).exists(), f"modality_config_path does not exist: {self.modality_config_path}"

        if isinstance(self.pov_cam_name_sim, str):
            self.pov_cam_name_sim = [self.pov_cam_name_sim]

        assert self.gr00t_embodiment_adapter is not None, (
            "gr00t_embodiment_adapter is required (it replaces task_mode_name): one of"
            f" {sorted(GR00T_EMBODIMENT_ADAPTER_CLASS_PATHS)} or the dotted class path of a Gr00tEmbodimentAdapter"
        )
        # Import the adapter class now so a bad name fails at config load, before any env is built.
        import_gr00t_embodiment_adapter_class(self.gr00t_embodiment_adapter)
