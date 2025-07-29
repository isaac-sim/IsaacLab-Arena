# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from abc import ABC, abstractmethod
from typing import Any

from isaac_arena.geometry.pose import Pose


class SceneBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_robot_initial_pose(self) -> Pose:
        raise NotImplementedError

    @abstractmethod
    def get_scene_cfg(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_observation_cfg(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_events_cfg(self) -> Any:
        raise NotImplementedError
