# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit one command to a running robot tool server and print its response."""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

COMMANDS = ("observe", "move_to", "set_gripper", "wait", "reset", "shutdown")


def _finite_number(value: str) -> float:
    """Parse a finite numeric command argument."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def _positive_number(value: str) -> float:
    """Parse a positive timeout."""
    number = _finite_number(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _positive_steps(value: str) -> int:
    """Parse a positive number of simulation steps."""
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if steps <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return steps


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--session_dir",
        type=Path,
        required=True,
        help="Directory of the running robot tool server.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_number,
        default=180.0,
        help="Response timeout in seconds (default: 180).",
    )
    parser.add_argument("--position", nargs=3, type=_finite_number, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--quaternion",
        nargs=4,
        type=_finite_number,
        metavar=("X", "Y", "Z", "W"),
        help="Target orientation in xyzw order; omit to preserve the current orientation.",
    )
    parser.add_argument(
        "--gripper",
        type=_finite_number,
        choices=(0.0, 1.0),
        help="Binary gripper target.",
    )
    parser.add_argument(
        "--steps",
        type=_positive_steps,
        help="Simulation step budget; omit to use the server default.",
    )
    parser.add_argument("--note", help="Short description saved with this command.")
    arguments = parser.parse_args()

    if arguments.command == "move_to" and arguments.position is None:
        parser.error("move_to requires --position X Y Z")
    if arguments.command == "set_gripper" and arguments.gripper is None:
        parser.error("set_gripper requires --gripper 0 or 1")
    if arguments.command != "move_to" and (arguments.position is not None or arguments.quaternion is not None):
        parser.error("--position and --quaternion are only valid for move_to")
    if arguments.gripper is not None and arguments.command not in (
        "move_to",
        "set_gripper",
    ):
        parser.error("--gripper is only valid for move_to and set_gripper")
    if arguments.steps is not None and arguments.command not in (
        "move_to",
        "set_gripper",
        "wait",
    ):
        parser.error("--steps is only valid for move_to, set_gripper, and wait")
    if arguments.quaternion is not None and not math.isclose(math.hypot(*arguments.quaternion), 1.0, abs_tol=1e-3):
        parser.error("--quaternion must have unit length within 0.001")
    return arguments


def _submit_request(request_directory: Path, request: dict) -> None:
    """Publish a complete request atomically for the server to consume."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=request_directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as request_file:
            json.dump(request, request_file, allow_nan=False)
            request_file.write("\n")
            request_file.flush()
            os.fsync(request_file.fileno())
        temporary_path.replace(request_directory / f"{request['id']}.json")
    finally:
        temporary_path.unlink(missing_ok=True)


def _wait_for_response(response_path: Path, timeout: float) -> dict:
    """Wait for the server response without retrying the command."""
    deadline = time.monotonic() + timeout
    while True:
        if response_path.is_file():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(response, dict) or type(response.get("ok")) is not bool:
                raise ValueError(f"Invalid server response in {response_path}: expected an object with boolean 'ok'")
            if "id" in response and response["id"] != response_path.stem:
                raise ValueError(f"Response request ID does not match {response_path.stem}")
            return response
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError(f"No response within {timeout:g} seconds")
        time.sleep(min(0.1, remaining_seconds))


def main() -> int:
    """Run one command and return zero only when the server reports success."""
    arguments = _parse_arguments()
    session_directory = arguments.session_dir.expanduser().resolve()
    request_directory = session_directory / "requests"
    response_directory = session_directory / "responses"
    request = {"id": uuid.uuid4().hex, "command": arguments.command}
    for field_name in ("position", "quaternion", "gripper", "steps", "note"):
        field_value = getattr(arguments, field_name)
        if field_value is not None:
            request[field_name] = field_value
    request_submitted = False
    try:
        if not (session_directory / "ready.json").is_file():
            raise FileNotFoundError(f"Server is not ready: {session_directory / 'ready.json'} is missing")
        if not request_directory.is_dir() or not response_directory.is_dir():
            raise FileNotFoundError(f"Server request and response directories are missing under {session_directory}")
        _submit_request(request_directory, request)
        request_submitted = True
        response = _wait_for_response(response_directory / f"{request['id']}.json", arguments.timeout)
        print(json.dumps(response, indent=2, allow_nan=False))
        return 0 if response["ok"] else 1
    except (OSError, ValueError, KeyboardInterrupt) as error:
        error_message = str(error) or "Interrupted while waiting for the server"
        failure = {"ok": False, "id": request["id"], "error": error_message}
        if request_submitted:
            failure["warning"] = (
                "The command may still execute. The client did not cancel or retry it. "
                "Check the response file before submitting another command."
            )
            failure["response_path"] = str(response_directory / f"{request['id']}.json")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
