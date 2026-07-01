# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

import websockets.sync.client
from openpi_client import msgpack_numpy, websocket_client_policy
from typing_extensions import override


class WebsocketClientPolicy(websocket_client_policy.WebsocketClientPolicy):
    """openpi WebsocketClientPolicy with configurable keepalive ping."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        *,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
    ) -> None:
        # Stored before super().__init__ because the base constructor calls
        # _wait_for_server(), which reads these back to open the connection.
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        super().__init__(host=host, port=port, api_key=api_key)

    @override
    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)
