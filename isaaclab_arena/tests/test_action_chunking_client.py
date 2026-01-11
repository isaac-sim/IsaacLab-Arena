# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026,
# The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import threading
import time
from typing import Any

import pytest

from isaaclab_arena.remote_policy.action_protocol import ActionProtocol, ChunkingActionProtocol
from isaaclab_arena.remote_policy.policy_server import PolicyServer
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
NUM_STEPS = 2
HOST = "127.0.0.1"
PORT = 5563  # test-only port, avoid conflicts


# ======================================================================================
# Dummy server-side policy using the real ChunkingActionProtocol
# ======================================================================================


class _DummyChunkingServerPolicy(ServerSidePolicy):
    """Server-side policy that uses ChunkingActionProtocol and returns fixed chunks."""

    def __init__(self, action_dim: int = 50, chunk_length: int = 4) -> None:
        super().__init__(config=None)
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._counter = 0

    def _build_protocol(self) -> ActionProtocol:
        return ChunkingActionProtocol(
            action_dim=self._action_dim,
            observation_keys=["policy.robot_joint_pos"],
            action_chunk_length=self._chunk_length,
            action_horizon=self._chunk_length,
        )

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return (batch, chunk_length, action_dim) array with a simple pattern.

        The options argument is accepted to match PolicyServer._handle_get_action,
        but is not used in this dummy implementation.
        """
        first_key = next(iter(observation.keys()))
        batch = int(np.shape(observation[first_key])[0])

        base_value = float(self._counter)
        self._counter += 1

        chunk = np.full(
            (batch, self._chunk_length, self._action_dim),
            fill_value=base_value,
            dtype=np.float32,
        )
        # IMPORTANT: return a dict containing "action" and "info"
        return {"action": chunk}, {}

    # NEW: match what PolicyServer._handle_reset expects
    def reset(self, env_ids: list[int] | None = None, reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset policy state for the given environment ids.

        The implementation here is trivial; it just returns an OK status
        and does not keep any per-env state.
        """
        return {"status": "ok"}

    @staticmethod
    def add_args_to_parser(parser: Any) -> Any:
        return parser

    @staticmethod
    def from_args(args: Any) -> _DummyChunkingServerPolicy:
        return _DummyChunkingServerPolicy()


# ======================================================================================
# Helper to start/stop a PolicyServer in background
# ======================================================================================


@pytest.fixture
def running_dummy_chunking_server() -> PolicyServer:
    """Start a PolicyServer with _DummyChunkingServerPolicy on localhost."""
    policy = _DummyChunkingServerPolicy(chunk_length=4)
    server = PolicyServer(
        policy=policy,
        host=HOST,
        port=PORT,
        api_token=None,
        timeout_ms=2_000,
    )

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Give the server a short time to bind and start.
    time.sleep(0.2)

    try:
        yield server
    finally:
        # Ask the server to stop and wait for the thread.
        server.running = False
        thread.join(timeout=5.0)

        if hasattr(server, "close"):
            server.close()
        assert not thread.is_alive()


# ======================================================================================
# Helper to call policy_runner (same style as existing tests)
# ======================================================================================


def _run_policy_runner_with_action_chunking_client() -> None:
    """Run policy_runner.py with ActionChunkingClientSidePolicy in remote mode.

    The remote host/port are set to the dummy server started by the fixture.
    """
    args: list[str] = [
        TestConstants.python_path,
        f"{TestConstants.evaluation_dir}/policy_runner.py",
    ]

    args.extend([
        "--policy_type",
        "isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy",
    ])

    args.extend([
        "--remote_host",
        HOST,
        "--remote_port",
        str(PORT),
        "--remote_kill_on_exit",
    ])

    args.extend(["--num_steps", str(NUM_STEPS)])
    if HEADLESS:
        args.append("--headless")

    args.append("galileo_g1_locomanip_pick_and_place")
    args.extend(["--embodiment", "g1_wbc_joint"])
    args.extend(["--object", "brown_box"])

    run_subprocess(args)


# ======================================================================================
# Test
# ======================================================================================


def test_action_chunking_client_end_to_end_with_dummy_chunking_server(
    running_dummy_chunking_server: PolicyServer,
) -> None:
    """End-to-end test: dummy chunking server + ActionChunkingClientSidePolicy + policy_runner.

    This verifies that:
    - The dummy PolicyServer using ChunkingActionProtocol can be reached on HOST:PORT.
    - ActionChunkingClientSidePolicy can connect to it via policy_runner.py.
    - The process exits successfully for a short rollout.
    """
    _run_policy_runner_with_action_chunking_client()
