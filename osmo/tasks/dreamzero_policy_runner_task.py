# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DreamZero policy-runner task for the Isaac Lab Arena OSMO workflow."""

from dataclasses import dataclass

from osmo.tasks.policy_runner_task import PolicyRunnerTask, PolicyRunnerTaskCfg
from osmo.workflows.utils.workflow_id import is_valid_workflow_id
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT

SERVER_WAIT_ATTEMPTS = 240
SERVER_WAIT_INTERVAL_SECONDS = 15
INITIAL_CONNECT_WAIT_SECONDS = 3600


@dataclass
class DreamZeroPolicyRunnerTaskCfg(PolicyRunnerTaskCfg):
    """Config for the DreamZero policy-runner task."""

    image: str = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:dreamzero-runner-20260709"
    """Arena image built with the isaaclab_arena_dreamzero package and the OSMO CLI
    (docker/push_to_ngc.sh -p -t <tag>); the shared :latest image has neither."""

    osmo_token_credential: str = "osmo-token"
    """GENERIC OSMO credential whose 'login_yaml_b64' field holds a base64 OSMO login.yaml."""


class DreamZeroPolicyRunnerTask(PolicyRunnerTask):
    """Policy-runner task that reaches its DreamZero server through an OSMO port-forward tunnel.

    The DreamZero server needs an H100 while Isaac Sim rendering needs an RTX-capable GPU
    (L40/L40S), and no OSMO pool offers both platforms; the two therefore run as separate
    workflows in separate pools (see ``workflows/dreamzero_split_workflows.py``). This task
    tunnels to the server workflow's task with ``osmo workflow port-forward`` (the CLI is
    baked into the policy-runner image) and points the policy at ``localhost``.
    Authentication reuses a base64-encoded OSMO ``login.yaml`` (which carries a refresh
    token, so the in-task CLI renews its session itself) injected through the GENERIC
    workflow credential named by the ``osmo_token_credential`` config field.

    On exit — success or failure — the task cancels the server workflow so the pair
    finishes together. This is best-effort: a hard-killed runner (OOM kill, node loss)
    skips the exit trap and leaves the server running until its own exec timeout.
    ``server_workflow_id`` is supplied by ``DreamZeroEvaluationWorkflow``, which owns the
    server's submission.
    """

    def __init__(
        self,
        task_cfg: DreamZeroPolicyRunnerTaskCfg,
        server_workflow_id: str,
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg, lead=lead)
        # This ID is spliced unquoted into the generated bash entry script, so reject any
        # value that could word-split or inject there (see is_valid_workflow_id).
        assert is_valid_workflow_id(server_workflow_id), f"Invalid OSMO workflow ID: {server_workflow_id!r}"
        self.server_workflow_id = server_workflow_id

    def _get_credentials(self) -> dict[str, dict[str, str]]:
        return {
            **super()._get_credentials(),
            self.task_cfg.osmo_token_credential: {"OSMO_LOGIN_YAML_B64": "login_yaml_b64"},
        }

    def _get_run_script(self) -> str:
        """Install the OSMO session, keep a tunnel to the server workflow alive, then run.

        The tunnel is supervised in a background loop so transient drops during a long
        rollout re-establish automatically; the policy client's own reconnect logic
        handles the WebSocket layer on top. The wait loop below only confirms the local
        tunnel listener — the tunnel accepts TCP before the remote server is up, so true
        server readiness is handled by the policy's retrying initial connect
        (``DreamZeroRemotePolicyConfig.initial_connect_wait_s``).
        """
        return (
            "set -euo pipefail\n"
            # On exit (success or failure): cancel the server workflow so the pair always
            # finishes together, then kill the background tunnel supervisor AND the
            # port-forward child it spawned (which outlives its parent as an orphan and
            # holds the task's output pipes open) or the task pod lingers as RUNNING —
            # but never `kill 0`, which would take the entry shell down too and turn a
            # successful run into a FAILED task.
            f"trap 'timeout 60 osmo workflow cancel {self.server_workflow_id}"
            ' --message "policy runner exited" || true;'
            " kill $(jobs -p) 2>/dev/null || true;"
            ' pkill -f "workflow port-forward" 2>/dev/null || true\' EXIT\n'
            'mkdir -p "$HOME/.config/osmo"\n'
            'printf \'%s\' "$OSMO_LOGIN_YAML_B64" | base64 -d > "$HOME/.config/osmo/login.yaml"\n'
            "set -x\n"
            "(\n"
            "  while true; do\n"
            f"    osmo workflow port-forward {self.server_workflow_id} {self.get_server_task_name()}"
            f" --port {POLICY_SERVER_PORT} || true\n"
            '    echo "Tunnel to DreamZero server exited; restarting in'
            f' {SERVER_WAIT_INTERVAL_SECONDS}s ..."\n'
            f"    sleep {SERVER_WAIT_INTERVAL_SECONDS}\n"
            "  done\n"
            ") &\n"
            f"for _ in $(seq 1 {SERVER_WAIT_ATTEMPTS}); do\n"
            f"  if (exec 3<>/dev/tcp/localhost/{POLICY_SERVER_PORT}) 2>/dev/null; then break; fi\n"
            f'  echo "Waiting for tunnel to DreamZero server workflow {self.server_workflow_id} ..."\n'
            f"  sleep {SERVER_WAIT_INTERVAL_SECONDS}\n"
            "done\n"
            f"{self._get_policy_runner_command()}\n"
        )

    def _get_policy_args(self) -> list[str]:
        return [
            "--policy_type",
            "isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy",
            "--remote_host",
            "localhost",
            "--remote_port",
            str(POLICY_SERVER_PORT),
            # The tunnel accepts TCP before the server is reachable through it, and the
            # server may still be loading its checkpoint; give the first connect a long
            # budget instead of the library's fail-fast default.
            "--initial_connect_wait_s",
            str(INITIAL_CONNECT_WAIT_SECONDS),
        ]

    @staticmethod
    def get_server_task_name() -> str:
        """Name of the server task inside the server workflow this runner tunnels to."""
        from osmo.tasks.dreamzero_server_task import DreamZeroServerTask

        return DreamZeroServerTask.get_task_name()
