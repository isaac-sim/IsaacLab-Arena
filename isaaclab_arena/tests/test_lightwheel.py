# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from isaaclab_arena.assets.sources.lightwheel import acquire_lightwheel_asset


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


class _FakeApiException(Exception):
    """Mimics lightwheel_sdk's ApiException message format: '<url> <status> <body>'."""

    def __init__(self, status: int, url: str = "https://api-dev.lightwheel.net//floorplan/v1/usd/get"):
        super().__init__(f"{url} {status} <html>gateway</html>")


def _make_flaky_loader(failures: list[Exception]):
    class Loader:
        client = None

        def __init__(self):
            self.calls = 0

        def acquire(self, **kwargs):
            self.calls += 1
            if self.calls <= len(failures):
                raise failures[self.calls - 1]
            return "ok"

    return Loader()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_acquire_lightwheel_asset_retries_transient_http_status(status):
    loader = _make_flaky_loader([_FakeApiException(status)])

    result = acquire_lightwheel_asset(loader, loader.acquire, "test asset", attempts=2, delay_sec=0)

    assert result == "ok"
    assert loader.calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_acquire_lightwheel_asset_does_not_retry_client_errors(status):
    loader = _make_flaky_loader([_FakeApiException(status)])

    with pytest.raises(_FakeApiException):
        acquire_lightwheel_asset(loader, loader.acquire, "test asset", attempts=3, delay_sec=0)

    assert loader.calls == 1


def test_acquire_lightwheel_asset_retries_status_without_url_prefix():
    # The SDK raises ApiException(response) with an empty path for the file download, giving ' 504 <body>'.
    loader = _make_flaky_loader([Exception(" 504 <html>gateway</html>")])

    assert acquire_lightwheel_asset(loader, loader.acquire, "test asset", attempts=2, delay_sec=0) == "ok"


def test_acquire_lightwheel_asset_retries_wrapped_connection_error():
    class ConnectionError(Exception):
        pass

    wrapper = Exception("UnknownException")
    wrapper.__cause__ = ConnectionError("connection reset by peer")
    loader = _make_flaky_loader([wrapper])

    assert acquire_lightwheel_asset(loader, loader.acquire, "test asset", attempts=2, delay_sec=0) == "ok"


def test_acquire_lightwheel_asset_gives_up_after_all_attempts():
    loader = _make_flaky_loader([_FakeApiException(504)] * 5)

    with pytest.raises(_FakeApiException):
        acquire_lightwheel_asset(loader, loader.acquire, "test asset", attempts=3, delay_sec=0)

    assert loader.calls == 3


def test_acquire_lightwheel_asset_backs_off_exponentially(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("isaaclab_arena.assets.sources.lightwheel.time.sleep", sleeps.append)
    loader = _make_flaky_loader([_FakeApiException(504)] * 3)

    acquire_lightwheel_asset(loader, loader.acquire, "test asset", attempts=4, delay_sec=5)

    assert sleeps == [5, 10, 20]
