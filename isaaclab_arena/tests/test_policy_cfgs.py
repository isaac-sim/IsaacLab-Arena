# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed policy configuration contract."""

from dataclasses import dataclass, is_dataclass

import pytest
import torch

from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy, ReplayActionPolicyCfg
from isaaclab_arena.policy.rsl_rl_action_policy import RslRlActionPolicy, RslRlActionPolicyCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg


@pytest.mark.parametrize(
    ("policy_type", "cfg_type"),
    [
        (ZeroActionPolicy, ZeroActionPolicyCfg),
        (ReplayActionPolicy, ReplayActionPolicyCfg),
        (RslRlActionPolicy, RslRlActionPolicyCfg),
    ],
)
def test_core_policies_declare_typed_configs(policy_type, cfg_type):
    """Check that each registered core policy declares its concrete config."""
    assert policy_type.config_class is cfg_type
    assert issubclass(cfg_type, PolicyCfg)
    assert is_dataclass(cfg_type)


@dataclass
class _ExamplePolicyCfg(PolicyCfg):
    value: int = 1


class _ExamplePolicy(PolicyBase[_ExamplePolicyCfg]):
    config_class = _ExamplePolicyCfg

    def get_action(self, env, observation) -> torch.Tensor:
        return torch.tensor([self.config.value])


def test_policy_base_constructs_from_typed_config_without_cli_methods():
    """Keep argparse adapters outside the core policy contract."""
    policy = _ExamplePolicy.from_dict({"value": 2})

    assert policy.config == _ExamplePolicyCfg(value=2)
    assert not hasattr(PolicyBase, "add_args_to_parser")
    assert not hasattr(PolicyBase, "from_args")
