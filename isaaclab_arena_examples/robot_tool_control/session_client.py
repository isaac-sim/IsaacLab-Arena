# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect a session policy request or submit one action chunk without importing Isaac Sim."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

IDENTITY_FIELDS = ("protocol_version", "policy_instance_id", "env_id", "episode_index", "request_id")


def inspect_session(session_directory: Path) -> dict:
    """Return session metadata and its outstanding request, if present."""
    session_directory = session_directory.expanduser().resolve()
    manifest = json.loads((session_directory / "session.json").read_text(encoding="utf-8"))
    inspection = {
        "session_directory": str(session_directory),
        "session": manifest,
        "controller_prompt_path": str(session_directory / manifest["controller_prompt_path"]),
    }
    if manifest["active_request"] is not None:
        request_path = session_directory / manifest["active_request"]
        inspection["request_path"] = str(request_path)
        inspection["request"] = json.loads(request_path.read_text(encoding="utf-8"))
    return inspection


def submit_actions(request_path: Path, actions: list, note: str | None = None) -> Path:
    """Atomically submit one response, refusing duplicate or inactive requests."""
    if not isinstance(actions, list):
        raise ValueError("The actions file must contain a JSON array of action rows")
    request_path = request_path.expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    identity = {name: request[name] for name in IDENTITY_FIELDS}
    if type(identity["protocol_version"]) is not int or identity["protocol_version"] != 1:
        raise ValueError("Unsupported session exchange protocol version")
    # Requests have a fixed layout beneath the session; no host or container paths are stored in the envelope.
    session_directory = request_path.parents[4]
    session = json.loads((session_directory / "session.json").read_text(encoding="utf-8"))
    request_state = json.loads((request_path.parent / "request_state.json").read_text(encoding="utf-8"))
    if session["policy_instance_id"] != identity["policy_instance_id"]:
        raise ValueError("Request belongs to a different policy instance")
    if (
        session["status"] != "open"
        or session["active_request"] != str(request_path.relative_to(session_directory))
        or request_state["status"] != "pending"
    ):
        raise ValueError("Request is no longer awaiting a response")
    response = {**identity, "actions": actions}
    if note is not None:
        response["note"] = note
    response_path = request_path.parent / "response.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=request_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as response_file:
            json.dump(response, response_file, indent=2, allow_nan=False)
            response_file.write("\n")
            response_file.flush()
            os.fsync(response_file.fileno())
        # A hard link publishes the complete file atomically and fails if another response already exists.
        os.link(temporary_path, response_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return response_path


def main() -> int:
    """Run an inspection or submission command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Show the outstanding request and session metadata")
    inspect_parser.add_argument("--session", type=Path, required=True)
    submit_parser = commands.add_parser("submit", help="Respond once to the given request")
    submit_parser.add_argument("--request", type=Path, required=True)
    submit_parser.add_argument(
        "--actions", type=Path, required=True, help="JSON file containing an H-by-8 action array"
    )
    submit_parser.add_argument("--note", help="Optional explanation saved with the response")
    arguments = parser.parse_args()
    try:
        if arguments.command == "inspect":
            result = inspect_session(arguments.session)
        else:
            actions = json.loads(arguments.actions.read_text(encoding="utf-8"))
            response_path = submit_actions(arguments.request, actions, arguments.note)
            result = {"response_path": str(response_path)}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
