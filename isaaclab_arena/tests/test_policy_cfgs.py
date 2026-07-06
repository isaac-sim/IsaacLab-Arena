# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed policy configuration contract."""

import argparse
import torch
from dataclasses import dataclass

import pytest

import isaaclab_arena.assets.register as policy_registration
from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_cli_args, build_policy_from_cli, policy_cfg_from_cli
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy, ReplayActionPolicyCfg
from isaaclab_arena.policy.rsl_rl_action_policy import RslRlActionPolicy, RslRlActionPolicyCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg

# TODO(cvolk, 2026-07-03): Delete the CLI compatibility tests when policy_runner
# receives typed policy configs directly.


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


def test_policy_registration_resolves_policy_from_concrete_config():
    """Resolve runtime policy construction from an experiment's concrete config."""
    assert PolicyRegistry().get_policy_type_for_cfg(ZeroActionPolicyCfg()) is ZeroActionPolicy


@dataclass
class _ExamplePolicyCfg(PolicyCfg):
    value: int = 1


class _ExamplePolicy(PolicyBase[_ExamplePolicyCfg]):
    name = "example_policy"

    def get_action(self, env, observation) -> torch.Tensor:
        return torch.tensor([self.config.value])


def _policy_parser(policy_type: type[PolicyBase]) -> argparse.ArgumentParser:
    """Create the shared runner parser and add one policy's generated flags."""
    parser = get_isaaclab_arena_cli_parser()
    return add_policy_cli_args(parser, policy_type)


def test_policy_runtime_contract_accepts_typed_config():
    """Construct and run a policy using only its typed runtime config."""
    config = _ExamplePolicyCfg(value=2)
    policy = _ExamplePolicy(config)

    assert policy.config is config
    assert policy.get_action(env=None, observation=None).item() == 2


@pytest.mark.parametrize(
    ("policy_type", "cli_args", "expected_cfg"),
    [
        (ZeroActionPolicy, [], ZeroActionPolicyCfg()),
        (
            ReplayActionPolicy,
            ["--replay_file_path", "episode.hdf5", "--device", "cpu", "--episode_name", "episode_1"],
            ReplayActionPolicyCfg(replay_file_path="episode.hdf5", device="cpu", episode_name="episode_1"),
        ),
        (
            RslRlActionPolicy,
            ["--checkpoint_path", "model.pt"],
            RslRlActionPolicyCfg(checkpoint_path="model.pt"),
        ),
    ],
)
def test_core_policy_cli_flags_reconstruct_registered_cfg(policy_type, cli_args, expected_cfg):
    """Generate core policy flags and reconstruct the registered config type."""
    parser = _policy_parser(policy_type)

    policy_cfg = policy_cfg_from_cli(policy_type, parser.parse_args(cli_args))

    assert policy_cfg == expected_cfg


def test_policy_cli_builds_registered_cfg(monkeypatch):
    """Build a policy from its registered config and a generated CLI override."""
    monkeypatch.setattr(
        PolicyRegistry,
        "get_policy_cfg_type",
        lambda self, policy_type: _ExamplePolicyCfg,
    )
    parser = _policy_parser(_ExamplePolicy)
    args_cli = parser.parse_args(["--value", "2"])

    policy = build_policy_from_cli(_ExamplePolicy, args_cli)

    assert policy.config == _ExamplePolicyCfg(value=2)


def test_shared_policy_cli_default_must_match_cfg():
    """Reject drift between shared runner flags and policy config defaults."""
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=1)

    with pytest.raises(AssertionError, match="ReplayActionPolicyCfg.device defaults to 'cuda:0'"):
        add_policy_cli_args(parser, ReplayActionPolicy)


def test_registered_policies_do_not_own_argparse_adapters():
    """Keep argparse generation outside typed policy implementations."""
    for policy_type in (ZeroActionPolicy, ReplayActionPolicy, RslRlActionPolicy):
        assert "add_args_to_parser" not in policy_type.__dict__
        assert "from_args" not in policy_type.__dict__


def test_policy_registration_infers_config_from_generic(monkeypatch):
    """Register the concrete config declared by ``PolicyBase[Cfg]``."""
    registrations = []

    def record_registration(self, policy_type, cfg_type):
        registrations.append((policy_type, cfg_type))

    monkeypatch.setattr(PolicyRegistry, "is_registered", lambda self, name, ensure_loaded=False: False)
    monkeypatch.setattr(PolicyRegistry, "register_policy", record_registration)

    registered_policy = policy_registration.register_policy(_ExamplePolicy)

    assert registered_policy is _ExamplePolicy
    assert registrations == [(_ExamplePolicy, _ExamplePolicyCfg)]


def test_unregistered_policy_config_is_rejected():
    """Require every policy to register a typed configuration."""
    with pytest.raises(AssertionError, match="Policy _ExamplePolicy must register a PolicyCfg"):
        PolicyRegistry().get_policy_cfg_type(_ExamplePolicy)
