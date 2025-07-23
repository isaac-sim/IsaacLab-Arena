# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from dataclasses import MISSING

from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.scene.scene import SceneBase
from isaac_arena.tasks.task import TaskBase
from isaaclab.utils import configclass

# TODO(alexmillane, 2025-07-23): For some reason, missing values are not being detected.
# We need to fix this as it's unsafe.


@configclass
class IsaacArenaEnvironment:
    """Describes an environment in Isaac Arena."""

    name: str = MISSING
    """The name of the environment."""

    embodiment: EmbodimentBase = MISSING
    """The embodiment to use in the environment."""

    scene: SceneBase = MISSING
    """The scene to use in the environment."""

    task: TaskBase = MISSING
    """The task to use in the environment."""
