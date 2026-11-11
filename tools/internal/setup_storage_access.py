# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sign in to NVIDIA Storage when needed and verify Arena asset access."""

from __future__ import annotations

import json
import os
import tempfile
import time
import toml
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
        # OAuth POSTs report pending and failed device logins as JSON error bodies.
        if values is None:
            raise
        return json.load(error)


def _read_auth_config() -> dict[str, Any]:
    """Read saved credentials, or return an empty config when the file is absent."""
    try:
        return toml.loads(_AUTH_PATH.read_text())
    except FileNotFoundError:
        return {}


def _save_refresh_token(refresh_token: str) -> None:
    """Save this stack's token atomically while preserving other configuration entries."""
    assert isinstance(refresh_token, str) and refresh_token.strip(), "Login returned no refresh token"
    config = _read_auth_config()
    stack_config = config.setdefault(_STORAGE_STACK, {})
    assert isinstance(stack_config, dict), f"Expected a table for {_STORAGE_STACK} in {_AUTH_PATH}"
    stack_config["refreshToken"] = refresh_token
    credentials = toml.dumps(config)
    _AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # NamedTemporaryFile creates an owner-only file in the destination directory.
        with tempfile.NamedTemporaryFile(mode="w", dir=_AUTH_PATH.parent, prefix=".auth-", delete=False) as output:
            temporary_path = Path(output.name)
            output.write(credentials)
        os.replace(temporary_path, _AUTH_PATH)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sign_in_if_needed() -> None:
    """Run device login unless this stack already has a nonempty refresh token."""
    stack_config = _read_auth_config().get(_STORAGE_STACK, {})
    assert isinstance(stack_config, dict), f"Expected a table for {_STORAGE_STACK} in {_AUTH_PATH}"
    refresh_token = stack_config.get("refreshToken")
    if isinstance(refresh_token, str) and refresh_token.strip():
        print(f"Using existing credentials for {_STORAGE_STACK} at {_AUTH_PATH}")
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
        _save_refresh_token(tokens["refresh_token"])
        print(f"Saved credentials to {_AUTH_PATH} (mode 600)")
        return
    raise RuntimeError("Device code expired; run setup again.")


def _verify_access() -> None:
    """List the Arena storage root and report write access."""
    import omni.client as client

    assert client.initialize(), "Failed to initialize omni.client"
    registration = None
    renewal_hint = (
        "If saved credentials are invalid or expired, renew the login using "
        "https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/"
        "-/blob/master/doc/user/kit/kit-vm-quickstart.md"
    )
    try:
        registration = client.register_storage_from_discovery(_STORAGE_STACK)
        assert (
            registration is not None and registration.result == client.Result.OK
        ), f"Storage registration failed. {renewal_hint}"
        url = f"{ARENA_STAGING_STORAGE_DIR}/Arena"
        result, entry = client.stat(url)
        assert result == client.Result.OK, f"Cannot access {url}: {result}. {renewal_hint}"
        print(f"Folder: {url}")
        print(f"Write access: {bool(entry.access & client.AccessFlags.WRITE)}")
        result, entries = client.list(url)
        assert result == client.Result.OK, f"Cannot list {url}: {result}. {renewal_hint}"
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
