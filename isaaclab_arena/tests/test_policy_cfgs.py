# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed policy configuration contract."""

import torch
from dataclasses import dataclass, is_dataclass

import pytest

import isaaclab_arena.assets.register as policy_registration
from isaaclab_arena.assets.registries import PolicyRegistry
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
def test_core_policies_register_typed_configs(policy_type, cfg_type):
    """Check that each core policy is registered with its concrete config."""
    assert PolicyRegistry().get_policy_cfg_type(policy_type) is cfg_type
    assert issubclass(cfg_type, PolicyCfg)
    assert is_dataclass(cfg_type)


@dataclass
class _ExamplePolicyCfg(PolicyCfg):
    value: int = 1


class _ExamplePolicy(PolicyBase[_ExamplePolicyCfg]):
    def get_action(self, env, observation) -> torch.Tensor:
        return torch.tensor([self.config.value])


def test_policy_constructs_without_deserialization_on_runtime_contract():
    """Keep deserialization and argparse adapters outside the policy contract."""
    policy = _ExamplePolicy(_ExamplePolicyCfg(value=2))

    assert policy.config == _ExamplePolicyCfg(value=2)
    assert not hasattr(PolicyBase, "config_class")
    assert not hasattr(PolicyBase, "from_dict")
    assert not hasattr(PolicyBase, "add_args_to_parser")
    assert not hasattr(PolicyBase, "from_args")


def test_bare_policy_registration_is_deprecated(monkeypatch):
    """Keep the untyped registration form only as an explicit compatibility path."""
    monkeypatch.setattr(policy_registration, "_register_policy", lambda policy_type, cfg_type: policy_type)

    with pytest.warns(DeprecationWarning, match="Bare @register_policy is deprecated"):
        registered_policy = policy_registration.register_policy(_ExamplePolicy)

    assert registered_policy is _ExamplePolicy
