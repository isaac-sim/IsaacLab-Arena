# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry binding each remote client policy to the OSMO inference server that serves it.

The Arena-experiment workflow derives, per Run, which server to co-schedule from the Run's
client policy config type. Each binding declares the server task to launch, the resource it
needs, how to point the client at its server, and any per-server compatibility checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicyCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.tasks.gr00t_server_task import Gr00tServerTask, Gr00tServerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

# Servers default to the same resource the workflow uses today; a server type that needs
# different hardware overrides these, and the workflow asserts all servers in one experiment agree.
_DEFAULT_SERVER_POOL = WorkflowCfg().pool
_DEFAULT_SERVER_PLATFORM = WorkflowCfg().platform


@dataclass(frozen=True)
class ServerBinding:
    """Bind one remote client policy to its inference server."""

    name: str
    """Server-type key: the ``servers.<name>`` config field and OSMO task-name stem."""

    server_task_cls: type[BaseTask]
    """OSMO task that serves this policy."""

    pool: str
    """OSMO pool this server must run in."""

    platform: str
    """Hardware platform this server must run on."""

    configure_client: Callable[[PolicyCfg, str, TaskCfg], None]
    """Point a Run's client policy at its dedicated server task (mutates the policy config)."""

    check: Callable[[list[ArenaRunCfg], TaskCfg], None]
    """Validate the Runs served by this server against the server config (no-op if none)."""


def _configure_pi0_client(policy_cfg: PolicyCfg, server_task_name: str, server_cfg: TaskCfg) -> None:
    assert isinstance(policy_cfg, Pi0RemotePolicyCfg)
    assert isinstance(server_cfg, Pi0ServerTaskCfg)
    policy_cfg.remote_host = Pi0ServerTask.host_token(server_task_name)
    policy_cfg.remote_port = POLICY_SERVER_PORT
    # The first OSMO inference may compile longer than the policy's normal keepalive timeout;
    # use the timeout owned by this server deployment.
    policy_cfg.ping_timeout = server_cfg.client_ping_timeout_s


def _check_pi0_variants(runs_using_binding: list[ArenaRunCfg], server_cfg: TaskCfg) -> None:
    assert isinstance(server_cfg, Pi0ServerTaskCfg)
    incompatible_policy_variants_by_run = {
        run_cfg.name: run_cfg.policy.policy_variant
        for run_cfg in runs_using_binding
        if run_cfg.policy.policy_variant != server_cfg.policy_variant
    }
    assert not incompatible_policy_variants_by_run, (
        f"pi0_remote Runs require variants {incompatible_policy_variants_by_run}, but the pi0 server is configured"
        f" for '{server_cfg.policy_variant}'"
    )


def _configure_gr00t_client(policy_cfg: PolicyCfg, server_task_name: str, server_cfg: TaskCfg) -> None:
    assert isinstance(policy_cfg, Gr00tRemoteClosedloopPolicyCfg)
    policy_cfg.remote_host = Gr00tServerTask.host_token(server_task_name)
    policy_cfg.remote_port = POLICY_SERVER_PORT


def _check_none(runs_using_binding: list[ArenaRunCfg], server_cfg: TaskCfg) -> None:
    """No per-server compatibility check."""


PI0_SERVER_BINDING = ServerBinding(
    name="pi0",
    server_task_cls=Pi0ServerTask,
    pool=_DEFAULT_SERVER_POOL,
    platform=_DEFAULT_SERVER_PLATFORM,
    configure_client=_configure_pi0_client,
    check=_check_pi0_variants,
)

GR00T_SERVER_BINDING = ServerBinding(
    name="gr00t",
    server_task_cls=Gr00tServerTask,
    pool=_DEFAULT_SERVER_POOL,
    platform=_DEFAULT_SERVER_PLATFORM,
    configure_client=_configure_gr00t_client,
    check=_check_none,
)

# Client policy config type -> the server that serves it. A Run whose policy type is absent
# here (e.g. a local zero-action policy) runs standalone with no server.
REMOTE_POLICY_SERVERS: dict[type[PolicyCfg], ServerBinding] = {
    Pi0RemotePolicyCfg: PI0_SERVER_BINDING,
    Gr00tRemoteClosedloopPolicyCfg: GR00T_SERVER_BINDING,
}

# Server-type name -> binding, for resolving the ``servers.<name>`` config field.
SERVER_BINDINGS_BY_NAME: dict[str, ServerBinding] = {
    binding.name: binding for binding in REMOTE_POLICY_SERVERS.values()
}


@dataclass
class ServersCfg:
    """Per-server-type deployment config. Field names match each binding's ``name``.

    Every server type is always present with defaults so its ``servers.<name>.*`` overrides
    compose under Hydra (a ``None`` sub-config would carry no schema). Which servers actually
    launch is decided per Run from the Run's client policy type, not from these fields.
    """

    pi0: Pi0ServerTaskCfg = field(default_factory=Pi0ServerTaskCfg)
    gr00t: Gr00tServerTaskCfg = field(default_factory=Gr00tServerTaskCfg)
