# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type

from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena.remote_policy.policy_server import PolicyServer
from isaaclab_arena.remote_policy.policy_registry import policy_registry


def resolve_policy_class(policy_type: str) -> Type[ServerSidePolicy]:
    return policy_registry.resolve_class(policy_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("IsaacLab Arena Remote Policy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api_token", type=str, default=None)
    parser.add_argument("--timeout_ms", type=int, default=5000)

    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        choices=policy_registry.available_policy_types(),
        help="Which remote policy to run (e.g. 'gr00t_closedloop').",
    )
    parser.add_argument(
        "--policy_config_yaml_path",
        type=str,
        required=True,
        help="Path to policy-specific config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    policy_cls = resolve_policy_class(args.policy_type)
    policy = policy_cls(policy_config_yaml_path=Path(args.policy_config_yaml_path))

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

