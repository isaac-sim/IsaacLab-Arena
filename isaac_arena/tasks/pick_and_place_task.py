# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import isaaclab.envs.mdp as mdp_isaac_lab
from isaac_arena.tasks.task import TaskBase
from isaac_arena.tasks.terminations.object_in_drawer import object_in_drawer
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass


class PickAndPlaceTask(TaskBase):
    def __init__(self):
        super().__init__()

    def get_termination_cfg(self):
        return TerminationsCfg()

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # TODO(cvolk): Make this config generic and move instance out.
    # time_out: TerminationTermCfg = MISSING
    # termination_terms: TerminationTermCfg = MISSING
    # success: TerminationTermCfg = MISSING
    time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # TODO(alex.millane, 2025.07.22): This is specific to the drawer scene. Make this generic.
    object_dropped = TerminationTermCfg(
        func=mdp_isaac_lab.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("pick_up_object")},
    )
    # TODO(alex.millane, 2025.07.22): This is specific to the drawer scene. Make this generic
    # to support other destination objects.
    success = TerminationTermCfg(func=object_in_drawer)
