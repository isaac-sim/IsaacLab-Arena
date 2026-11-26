# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record a lift relative to an object's resting pose in each environment."""

import math
import numpy as np
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.spatial import object_is_above_height
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class LiftObjectTask(TaskBase):
    """Require an object to rise above its first resting height in the episode.

    Success depends only on the object's height, so it does not check how the object was lifted.
    """

    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        minimum_lift_m: float = 0.1,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        """Configure a lift task relative to the episode's first rest pose.

        Args:
            lift_object: Object to lift.
            background_scene: Background defining the drop-failure height.
            minimum_lift_m: Required height increase in meters, strictly positive.
            episode_length_s: Maximum episode duration in seconds.
            task_description: Optional natural-language instruction.
        """
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert (
            math.isfinite(minimum_lift_m) and minimum_lift_m > 0
        ), f"minimum_lift_m must be finite and positive, got {minimum_lift_m}"
        self.lift_object = lift_object
        self.background_scene = background_scene
        self.minimum_lift_m = minimum_lift_m

    def apply_reachability_constraints(self) -> None:
        self._apply_reachability_constraints([self.lift_object])

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_viewer_cfg(self):
        return get_viewer_cfg_look_at_object(self.lift_object, offset=np.array([-1.5, -1.5, 1.5]))

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="lift_object",
                    predicate_sequence=[
                        # Records the object's first resting position as its lift reference.
                        partial(
                            object_is_above_height,
                            object_name=self.lift_object.name,
                            use_settled_state=True,
                            distance=self.minimum_lift_m,
                        ),
                    ],
                )
            ],
            failures={
                "object_dropped": TerminationTermCfg(
                    func=mdp.root_height_below_minimum,
                    params={
                        "minimum_height": self.background_scene.object_min_z,
                        "asset_cfg": SceneEntityCfg(self.lift_object.name),
                    },
                )
            },
        )
