# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import warnings
from dataclasses import fields
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.cli.dataclass_cli import (
    add_dataclass_cli_args,
    assert_cli_defaults_match_dataclass,
    dataclass_from_cli,
)

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


# Legacy argparse compatibility
#
# Registered policy configs are the source of truth. Until policy_runner receives a
# PolicyCfg directly, these helpers generate its policy flags and reconstruct the same
# config from the parsed Namespace. Untyped downstream policies temporarily keep their
# existing add_args_to_parser() and from_args() path.
# TODO(cvolk, 2026-07-03): Delete this compatibility section when policy_runner
# receives typed policy configs directly and untyped downstream policies are removed.
_FIELDS_PROVIDED_BY_SHARED_PARSER = {"device", "num_envs"}


def _add_policy_cfg_arguments(
    parser: argparse.ArgumentParser,
    policy_cfg_type: type["PolicyCfg"],
) -> None:
    """Generate CLI flags from one registered policy config."""
    cfg_field_names = {config_field.name for config_field in fields(policy_cfg_type)}
    shared_fields = cfg_field_names.intersection(_FIELDS_PROVIDED_BY_SHARED_PARSER)
    assert_cli_defaults_match_dataclass(parser, policy_cfg_type, shared_fields)
    add_dataclass_cli_args(parser, policy_cfg_type, excluded_fields=shared_fields)


def add_policy_cli_args(
    parser: argparse.ArgumentParser,
    policy_type: type["PolicyBase"],
) -> argparse.ArgumentParser:
    """Add generated policy flags or use an untyped policy's legacy adapter."""
    policy_cfg_type = PolicyRegistry().get_policy_cfg_type(policy_type)
    if policy_cfg_type is not None:
        _add_policy_cfg_arguments(parser, policy_cfg_type)
        return parser

    # TODO(cvolk, 2026-07-03): Delete this fallback when downstream policies register PolicyCfg.
    warnings.warn(
        f"Policy {policy_type.__name__} uses deprecated policy-owned argparse flags; register a PolicyCfg",
        DeprecationWarning,
        stacklevel=2,
    )
    return policy_type.add_args_to_parser(parser)


def policy_cfg_from_cli(
    policy_type: type["PolicyBase"],
    args_cli: argparse.Namespace,
) -> "PolicyCfg":
    """Create a registered policy's typed config from parsed CLI values."""
    policy_cfg_type = PolicyRegistry().get_policy_cfg_type(policy_type)
    assert policy_cfg_type is not None, f"Policy {policy_type.__name__} must register a PolicyCfg"
    return dataclass_from_cli(policy_cfg_type, args_cli)


def build_policy_from_cli(
    policy_type: type["PolicyBase"],
    args_cli: argparse.Namespace,
) -> "PolicyBase":
    """Build a policy through its typed or untyped legacy CLI path."""
    if PolicyRegistry().get_policy_cfg_type(policy_type) is not None:
        return policy_type(policy_cfg_from_cli(policy_type, args_cli))

    # TODO(cvolk, 2026-07-03): Delete this fallback when downstream policies register PolicyCfg.
    return policy_type.from_args(args_cli)


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy runner specific arguments to the parser."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use. This is either a registered policy name or a path to a policy class.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the policy (if num_episodes is not provided)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run the policy (if num_steps is not provided)",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy. Takes precedence over the task's own description.",
    )
    parser.add_argument(
        "--record_viewport_video",
        action="store_true",
        default=False,
        help="Record an mp4 video of the rollout viewport (uses gymnasium.wrappers.RecordVideo).",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/eval/output",
        help=(
            "Base directory for evaluation outputs (videos, per-episode results, report); a"
            " reverse-dated run subdirectory is added per run."
        ),
    )
    parser.add_argument(
        "--record_camera_video",
        action="store_true",
        default=False,
        help=(
            "Record one mp4 per camera in obs['camera_obs'] (what the policy actually sees)."
            " Independent of --record_viewport_video; use either or both."
        ),
    )
    parser.add_argument(
        "--serve_evaluation_report",
        action="store_true",
        default=False,
        help="After all jobs finish, serve the evaluation report over HTTP.",
    )
    parser.add_argument(
        "--evaluation_report_port",
        type=int,
        default=8000,
        help="Port to serve the evaluation report on when --serve_evaluation_report is set. Defaults to 8000.",
    )
