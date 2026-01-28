# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import objects_on_destinations, root_height_below_minimum_multi_objects
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class SortMultiObjectTask(TaskBase):

    def __init__(
        self,
        pick_up_object_list: list[Asset],
        destination_location_list: list[Asset],
        background_scene: Asset,
        episode_length_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.pick_up_object_list = pick_up_object_list
        self.destination_location_list = destination_location_list
        self.background_scene = background_scene

        assert len(pick_up_object_list) == len(destination_location_list)

        pick_up_object_contact_sensor_list = []
        for pick_up_object, destination_location in zip(pick_up_object_list, destination_location_list):
            pick_up_object_contact_sensor_list.append(
                    pick_up_object.get_contact_sensor_cfg(
                    contact_against_prim_paths=[destination_location.get_prim_path()]
                )
            )
        self.pick_up_object_contact_sensor_list = pick_up_object_contact_sensor_list
        self.contact_sensor_name_list = [f"contact_sensor_{i}" for i in range(len(self.pick_up_object_contact_sensor_list))]

        self.events_cfg = None
        self.scene_config = self.make_scene_cfg()
        self.termination_cfg = self.make_termination_cfg()

    def make_scene_cfg(self):
        self.scene_config = SceneCfg()

        for name, pick_up_object_contact_sensor in zip(self.contact_sensor_name_list, self.pick_up_object_contact_sensor_list):
            setattr(self.scene_config, name, pick_up_object_contact_sensor)
        return self.scene_config
    
    def get_scene_cfg(self):
        return self.scene_config


    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        object_cfg_list = [SceneEntityCfg(pick_up_object.name) for pick_up_object in self.pick_up_object_list]
        contact_sensor_cfg_list = [SceneEntityCfg(name) for name in self.contact_sensor_name_list]

        success = TerminationTermCfg(
            func=objects_on_destinations,
            params={
                "object_cfg_list": object_cfg_list,
                "contact_sensor_cfg_list": contact_sensor_cfg_list,
                "force_threshold": 1.0,
                "velocity_threshold": 0.1,
            },
        )
        object_dropped = TerminationTermCfg(
            func=root_height_below_minimum_multi_objects,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg_list": [SceneEntityCfg(pick_up_object.name) for pick_up_object in self.pick_up_object_list],
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object_list[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class SceneCfg:
    """
    Scene configuration for the pick and place task.
    Note: only support <4 objects. Need to figure out a more flexible method, like __post_init__()
    """

    contact_sensor_0: ContactSensorCfg = MISSING
    contact_sensor_1: ContactSensorCfg = MISSING

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING
