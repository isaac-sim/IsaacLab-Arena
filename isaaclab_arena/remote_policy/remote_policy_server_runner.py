# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type
from importlib import import_module

from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena.remote_policy.policy_server import PolicyServer


def get_policy_cls(policy_type: str) -> type["ServerSidePolicy"]:
    """Get the policy class for the given policy type name.

       it tries to dynamically import the policy class, treating
       the policy_type argument as a string representing the module path and class name.
    """
    print(f"Dynamically importing from path: {policy_type}")
    assert "." in policy_type, (
        "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
        f" {policy_type}"
    )
    # Dynamically import the class from the string path
    module_path, class_name = policy_type.rsplit(".", 1)
    module = import_module(module_path)
    policy_cls = getattr(module, class_name)
    return policy_cls


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
        help="Which remote policy to run (e.g. 'isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy').",
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

    policy_cls = get_policy_cls(args.policy_type)
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

