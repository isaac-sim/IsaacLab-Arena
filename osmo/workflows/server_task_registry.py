# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Look up the OSMO policy server associated with an Arena client policy."""

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.utils.singleton import SingletonMeta
from osmo.tasks.policy_server_task import PolicyServerTask

_SERVER_TASKS_LOADED = False


class ServerTaskRegistry(metaclass=SingletonMeta):
    """Map Arena client policy types to their OSMO server task types."""

    def __init__(self) -> None:
        self._server_types_by_policy_type: dict[type[PolicyBase], type[PolicyServerTask]] = {}

    def register(self, server_type: type[PolicyServerTask]) -> None:
        """Register a policy server under the client policy type it serves."""
        assert (
            server_type.policy_type not in self._server_types_by_policy_type
        ), f"Policy {server_type.policy_type.__name__} already has a policy server"
        self._server_types_by_policy_type[server_type.policy_type] = server_type

    def has_server_for_policy_type(self, policy_type: type[PolicyBase]) -> bool:
        """Return whether a policy server is already registered for ``policy_type``."""
        return policy_type in self._server_types_by_policy_type

    def get_server_type_for_policy_cfg(self, policy_cfg: PolicyCfg) -> type[PolicyServerTask] | None:
        """Get the server task for a client policy config, or None if it needs no server."""
        ensure_server_tasks_registered()
        policy_type = PolicyRegistry().get_policy_type_for_cfg(policy_cfg)
        return self._server_types_by_policy_type.get(policy_type)

    def get_all_server_types(self) -> list[type[PolicyServerTask]]:
        """Return every registered policy server task type."""
        ensure_server_tasks_registered()
        return list(self._server_types_by_policy_type.values())


def register_server_task(cls: type[PolicyServerTask]) -> type[PolicyServerTask]:
    """Decorator registering a policy server with the ServerTaskRegistry."""
    registry = ServerTaskRegistry()
    if registry.has_server_for_policy_type(cls.policy_type):
        print(f"WARNING: Policy server for {cls.policy_type.__name__} is already registered. Doing nothing.")
    else:
        registry.register(cls)
    return cls


def ensure_server_tasks_registered() -> None:
    """Import policy-server modules so their ``@register_server_task`` decorators run."""
    global _SERVER_TASKS_LOADED
    if _SERVER_TASKS_LOADED:
        return
    _SERVER_TASKS_LOADED = True
    import osmo.tasks.cosmos_server_task  # noqa: F401
    import osmo.tasks.gr00t_server_task  # noqa: F401
    import osmo.tasks.pi0_server_task  # noqa: F401
