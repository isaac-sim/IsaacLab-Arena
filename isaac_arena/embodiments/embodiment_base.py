from abc import ABC
from typing import Any

from isaac_arena.geometry.pose import Pose
from isaaclab.utils import configclass

# NOTE(alexmillane, 2025.07.25): Consider if we need these classes.


@configclass
class ActionsCfg:
    pass


@configclass
class ObservationsCfg:
    pass


@configclass
class EventCfg:
    pass


class EmbodimentBase(ABC):

    def __init__(self):
        self.scene_config: Any | None = None
        self.action_config: ActionsCfg | None = None
        self.observation_config: ObservationsCfg | None = None
        self.event_config: EventCfg | None = None

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        return self.observation_config

    def get_event_cfg(self) -> Any:
        return self.event_config

    def set_robot_initial_pose(self, pose: Pose):
        if self.scene_config is None or not hasattr(self.scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        self.scene_config.robot.init_state.pos = pose.position_xyz
        self.scene_config.robot.init_state.rot = pose.rotation_wxyz
