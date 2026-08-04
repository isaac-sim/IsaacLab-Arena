# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base task for OSMO policy servers derived from Arena client policies."""

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from osmo.tasks.base_task import BaseTask, TaskCfg


class PolicyServerTask(BaseTask):
    """OSMO server task associated with one Arena client policy type."""

    policy_type: type[PolicyBase]
    """Client policy class served by this task."""

    @classmethod
    def task_cfg_for_policy(cls, policy_cfg: PolicyCfg) -> TaskCfg:
        """Build this server's task config from a client policy config."""
        assert PolicyRegistry().get_policy_type_for_cfg(policy_cfg) is cls.policy_type
        return cls.task_cfg_type()
