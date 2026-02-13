# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import argparse
from importlib import import_module

from isaaclab_arena.remote_policy.policy_server import PolicyServer
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy


def get_policy_cls(policy_type: str) -> type[ServerSidePolicy]:
    """Dynamically import and return a ServerSidePolicy subclass.

    The policy_type argument must be a fully qualified Python path of the form:
        "package.subpackage.module.ClassName"
    """
    print(f"[remote_policy_server_runner] Importing server-side policy from: {policy_type}")
    if "." not in policy_type:
        raise ValueError(
            "policy_type must be a dotted Python import path of the form "
            "'module.submodule.ClassName', "
            f"got: {policy_type!r}"
        )
    module_path, class_name = policy_type.rsplit(".", 1)
    module = import_module(module_path)
    policy_cls = getattr(module, class_name)
    return policy_cls


def build_base_parser() -> argparse.ArgumentParser:
    """Build the base CLI parser for the remote policy server.

    This parser only contains arguments that are common to all server-side policies.
    Policy-specific arguments are added later by the selected ServerSidePolicy subclass.
    """
    parser = argparse.ArgumentParser("IsaacLab Arena Remote Policy Server")

    # Generic server options.
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api_token", type=str, default=None)
    parser.add_argument("--timeout_ms", type=int, default=5000)

    # Which ServerSidePolicy implementation to run.
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help=(
            "Dotted Python path of the server-side policy to run, e.g. "
            "'isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy'."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments in two stages.

    1) Parse only the base arguments to discover which policy class to use.
    2) Let that class extend the parser with its own arguments, then parse again.
    """
    # Stage 1: parse base args to get policy_type.
    base_parser = build_base_parser()
    base_args, _ = base_parser.parse_known_args()

    policy_cls = get_policy_cls(base_args.policy_type)
    print(f"[remote_policy_server_runner] Requested server-side policy: {base_args.policy_type} -> {policy_cls}")

    # Stage 2: build a fresh parser, extend it with policy-specific arguments, then parse fully.
    full_parser = build_base_parser()
    if not hasattr(policy_cls, "add_args_to_parser"):
        raise TypeError(
            f"Server-side policy class {policy_cls} must define a static 'add_args_to_parser(parser)' method."
        )
    full_parser = policy_cls.add_args_to_parser(full_parser)  # type: ignore[assignment]

    args = full_parser.parse_args()
    return args


def main() -> None:
    """Entry point for running a remote policy server.

    The script:
      1) Parses CLI arguments in two stages.
      2) Instantiates the requested ServerSidePolicy via its from_args() helper.
      3) Wraps it in a PolicyServer and starts the RPC loop.
    """
    args = parse_args()

    policy_cls = get_policy_cls(args.policy_type)
    if not hasattr(policy_cls, "from_args"):
        raise TypeError(f"Server-side policy class {policy_cls} must define a static 'from_args(args)' method.")

    # Construct the server-side policy from CLI arguments.
    policy = policy_cls.from_args(args)  # type: ignore[call-arg]

    # Start the RPC server.
    server = PolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
        timeout_ms=args.timeout_ms,
    )
    server.run()


if __name__ == "__main__":
    main()
