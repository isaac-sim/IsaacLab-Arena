# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed policy configuration contract."""

import torch
from dataclasses import dataclass

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


@dataclass
class _ExamplePolicyCfg(PolicyCfg):
    value: int = 1


class _ExamplePolicy(PolicyBase[_ExamplePolicyCfg]):
    name = "example_policy"

    def get_action(self, env, observation) -> torch.Tensor:
        return torch.tensor([self.config.value])


def test_policy_runtime_contract_accepts_typed_config():
    """Construct and run a policy using only its typed runtime config."""
    config = _ExamplePolicyCfg(value=2)
    policy = _ExamplePolicy(config)

    assert policy.config is config
    assert policy.get_action(env=None, observation=None).item() == 2


def test_typed_policy_registration_associates_config(monkeypatch):
    """Pass the policy and its config through the supported decorator form."""
    registrations = []

    def record_registration(policy_type, cfg_type):
        registrations.append((policy_type, cfg_type))
        return policy_type

    monkeypatch.setattr(policy_registration, "_register_policy", record_registration)

    registered_policy = policy_registration.register_policy(cfg_type=_ExamplePolicyCfg)(_ExamplePolicy)

    assert registered_policy is _ExamplePolicy
    assert registrations == [(_ExamplePolicy, _ExamplePolicyCfg)]


def test_bare_policy_registration_is_deprecated(monkeypatch):
    """Keep the untyped registration form only as an explicit compatibility path."""
    # TODO(cvolk, 2026-07-03): Delete this test when bare @register_policy support is removed.
    monkeypatch.setattr(policy_registration, "_register_policy", lambda policy_type, cfg_type: policy_type)

    with pytest.warns(DeprecationWarning, match="Bare @register_policy is deprecated"):
        registered_policy = policy_registration.register_policy(_ExamplePolicy)

    assert registered_policy is _ExamplePolicy
