# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type

from isaaclab_arena.remote_policy.model_policy import ModelPolicy
from isaaclab_arena.remote_policy.policy_server import PolicyServer

POLICY_REGISTRY: dict[str, str] = {
    # policy_type: "module_path:ClassName"
    "gr00t_closedloop": "isaaclab_arena_gr00t.gr00t_remote_policy:Gr00tRemoteModelPolicy",
}


def resolve_policy_class(policy_type: str) -> Type[ModelPolicy]:
    """Dynamically import and return the ModelPolicy subclass for the given policy_type."""
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy_type={policy_type!r}. "
            f"Available options: {sorted(POLICY_REGISTRY.keys())}"
        )

    spec = POLICY_REGISTRY[policy_type]
    try:
        module_path, class_name = spec.split(":")
    except ValueError:
        raise RuntimeError(
            f"Invalid registry entry for policy_type={policy_type!r}: {spec!r} "
            "(expected 'module_path:ClassName')"
        )

    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as exc:
        raise ImportError(
            f"Failed to import module '{module_path}' for policy_type={policy_type!r}. "
            "This usually means the corresponding policy package is not installed "
            "in the current server environment."
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_path}' does not define class '{class_name}' "
            f"for policy_type={policy_type!r}."
        ) from exc

    if not issubclass(cls, ModelPolicy):
        raise TypeError(
            f"Resolved class '{class_name}' from '{module_path}' is not a ModelPolicy "
            f"subclass (policy_type={policy_type!r})."
        )
    return cls


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
        choices=sorted(POLICY_REGISTRY.keys()),
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

