# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from isaaclab_arena.assets.lightwheel_utils import acquire_lightwheel_asset


def test_acquire_lightwheel_asset_fetches_object():
    loader_module = pytest.importorskip("lightwheel_sdk.loader")
    object_loader = loader_module.object_loader
    old_timeout = object_loader.client.base_timeout

    file_path, object_name, metadata = acquire_lightwheel_asset(
        object_loader,
        object_loader.acquire_by_registry,
        "microwave asset",
        attempts=2,
        delay_sec=0,
        registry_type="fixtures",
        file_name="Microwave039",
        file_type="USD",
    )

    assert Path(file_path).exists()
    assert object_name == "Microwave039"
    assert metadata["fileName"].startswith("Microwave039")
    assert object_loader.client.base_timeout == old_timeout


def test_acquire_lightwheel_asset_retries_timeout_failure():
    class Client:
        base_timeout = 10

    class Loader:
        def __init__(self):
            self.client = Client()
            self.calls = 0
            self.timeouts_seen = []

        def acquire(self, **kwargs):
            self.calls += 1
            self.timeouts_seen.append(self.client.base_timeout)
            if self.calls == 1:
                raise TimeoutError("read timed out")
            return kwargs["asset_name"]

    loader = Loader()

    result = acquire_lightwheel_asset(
        loader,
        loader.acquire,
        "test asset",
        attempts=2,
        timeout_sec=45,
        delay_sec=0,
        asset_name="foo",
    )

    assert result == "foo"
    assert loader.calls == 2
    assert loader.timeouts_seen == [45, 45]
    assert loader.client.base_timeout == 10
