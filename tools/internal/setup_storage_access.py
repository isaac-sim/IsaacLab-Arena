# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sign in to NVIDIA Storage when needed and verify Arena asset access."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from isaaclab_arena.assets.nucleus import ARENA_STAGING_STORAGE_DIR

_STORAGE_STACK = "https://rc.storage.omni.nvidia.com"
_AUTH_PATH = Path.home() / ".nvidia-omniverse/config/auth.toml"


def _request_json(url: str, values: dict[str, str] | None = None) -> dict[str, Any]:
    """Send an HTTP request and return its JSON response."""
    data = urllib.parse.urlencode(values).encode() if values else None
    try:
        with urllib.request.urlopen(url, data=data, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        return json.load(error)


def _sign_in_if_needed() -> None:
    """Run device login unless credentials already exist."""
    if _AUTH_PATH.exists():
        print(f"Using existing credentials at {_AUTH_PATH}")
        return

    config = _request_json(f"{_STORAGE_STACK}/api/v1/auth-config")
    client = config["clients"]["default"]
    oidc = _request_json(config["openid_configuration"])
    device = _request_json(
        oidc["device_authorization_endpoint"],
        {"client_id": client["client_id"], "scope": client["scope"]},
    )
    assert "error" not in device, device.get("error_description", "Device login failed")
    print(f"Open: {device['verification_uri']}  Code: {device['user_code']}", flush=True)

    interval = device.get("interval", 5)
    deadline = time.monotonic() + device["expires_in"]
    while time.monotonic() < deadline:
        time.sleep(interval)
        tokens = _request_json(
            oidc["token_endpoint"],
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client["client_id"],
                "device_code": device["device_code"],
            },
        )
        error = tokens.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        assert not error, tokens.get("error_description", error)
        _AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(_AUTH_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write(f"[{json.dumps(_STORAGE_STACK)}]\n")
            output.write(f'refreshToken = {json.dumps(tokens["refresh_token"])}\n')
        print(f"Saved credentials to {_AUTH_PATH} (mode 600)")
        return
    raise RuntimeError("Device code expired; run setup again.")


def _verify_access() -> None:
    """List the Arena storage root and report write access."""
    import omni.client as client

    assert client.initialize(), "Failed to initialize omni.client"
    registration = None
    try:
        registration = client.register_storage_from_discovery(_STORAGE_STACK)
        assert registration is not None and registration.result == client.Result.OK, "Storage registration failed"
        url = f"{ARENA_STAGING_STORAGE_DIR}/Arena"
        result, entry = client.stat(url)
        assert result == client.Result.OK, f"Cannot access {url}: {result}"
        print(f"Folder: {url}")
        print(f"Write access: {bool(entry.access & client.AccessFlags.WRITE)}")
        result, entries = client.list(url)
        assert result == client.Result.OK, f"Cannot list {url}: {result}"
        for child in entries:
            print(child.relative_path)
    finally:
        if registration is not None:
            registration.stop()
        client.shutdown()


def main() -> None:
    """Sign in if needed, then verify Arena storage access."""
    _sign_in_if_needed()
    _verify_access()


if __name__ == "__main__":
    main()
