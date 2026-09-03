# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import object_on_destination
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import SuccessMode, check_success, root_height_below_minimum_multi_objects
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass


@register_task
class SortMultiObjectTask(TaskBase):

    def __init__(
        self,
        pick_up_object_list: list[Asset],
        destination_location_list: list[Asset],
        background_scene: Asset,
        episode_length_s: float | None = None,
        force_threshold: float = 1.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert len(pick_up_object_list) == len(destination_location_list)

        self.pick_up_object_list = pick_up_object_list
        self.destination_location_list = destination_location_list
        self.background_scene = background_scene
        assert force_threshold >= 0.0, f"force_threshold must be non-negative, got {force_threshold}"
        self.force_threshold = force_threshold

        self.pick_up_object_contact_sensor_list = []
        self.contact_sensor_name_list = []
        for pick_up_object, destination in zip(pick_up_object_list, destination_location_list):
            self.pick_up_object_contact_sensor_list.append(
                pick_up_object.get_contact_sensor_cfg(contact_against_object=destination)
            )
            self.contact_sensor_name_list.append(f"contact_sensor_{pick_up_object.name}")

        self.events_cfg = None
        self.scene_config = self.make_scene_cfg()
        self.termination_cfg = self.make_termination_cfg()

    def make_scene_cfg(self):

        # Support variable number of contact sensors.
        fields: list[tuple[str, type, ContactSensorCfg]] = []
        for contact_sensor_name, contact_sensor_cfg in zip(
            self.contact_sensor_name_list, self.pick_up_object_contact_sensor_list
        ):
            fields.append((contact_sensor_name, type(contact_sensor_cfg), contact_sensor_cfg))
        SceneCfg = make_configclass("SceneCfg", fields)
        scene_cfg = SceneCfg()
        return scene_cfg

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        object_cfg_list = [SceneEntityCfg(pick_up_object.name) for pick_up_object in self.pick_up_object_list]
        destination_cfg_list = [
            SceneEntityCfg(destination_location.name) for destination_location in self.destination_location_list
        ]
        contact_sensor_cfg_list = [SceneEntityCfg(name) for name in self.contact_sensor_name_list]

        object_on_destination_terms = [
            TerminationTermCfg(
                func=object_on_destination,
                params={
                    "object_cfg": object_cfg,
                    "destination_cfg": destination_cfg,
                    "contact_sensor_cfg": contact_sensor_cfg,
                    "force_threshold": self.force_threshold,
                    "velocity_threshold": 0.1,
                },
            )
            for object_cfg, destination_cfg, contact_sensor_cfg in zip(
                object_cfg_list,
                destination_cfg_list,
                contact_sensor_cfg_list,
                strict=True,
            )
        ]
        success = TerminationTermCfg(
            func=check_success,
            params={"predicates": object_on_destination_terms, "mode": SuccessMode.ALL},
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
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING
