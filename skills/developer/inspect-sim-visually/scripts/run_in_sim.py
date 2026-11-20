# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run Python code in a running Isaac Sim through the isaacsim.code_editor.python_server extension.

Examples::

    python3 -I run_in_sim.py --port 18210 --file sim_helpers.py
    python3 -I run_in_sim.py --port 18210 --code 'capture_viewer("/tmp/view.png")' --wait_for_file /tmp/view.png

Exit codes: 0 success, 1 the code raised in the sim, 2 no server or no reply, 3 --wait_for_file timed out.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

NO_REPLY_HINT = (
    "The server answers only while the simulation updates Kit every step: launch with --viz kit and DISPLAY unset "
    "(not HEADLESS=1), or with camera sensors enabled. It also pauses while the environment is being built and while "
    "the runner is blocked, for example on a policy server. The request may still run once the server resumes."
)


def send_request(host: str, port: int, code: str, timeout_s: float) -> dict:
    """Send code to the server and return its JSON reply.

    The server executes the request once the client half-closes the connection, then replies and closes.
    """
    envelope = json.dumps({"code": code, "timeout": timeout_s})
    with socket.create_connection((host, port), timeout=timeout_s) as connection:
        connection.settimeout(timeout_s)
        connection.sendall(envelope.encode())
        connection.shutdown(socket.SHUT_WR)
        reply_chunks = []
        while chunk := connection.recv(65536):
            reply_chunks.append(chunk)
    return json.loads(b"".join(reply_chunks).decode())


def wait_for_file(path: Path, timeout_s: float) -> bool:
    """Return True once the file exists, is non-empty and its size is unchanged across two polls."""
    deadline = time.monotonic() + timeout_s
    previous_size = -1
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else 0
        if size > 0 and size == previous_size:
            return True
        previous_size = size
        time.sleep(0.1)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help="Port of the python_server extension.")
    parser.add_argument("--host", default="127.0.0.1")
    code_source = parser.add_mutually_exclusive_group()
    code_source.add_argument("--code", help="Python code to run. Read from stdin when neither --code nor --file.")
    code_source.add_argument("--file", type=Path, help="File whose Python code to run.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for the reply.")
    parser.add_argument(
        "--wait_for_file", type=Path, help="After a successful reply, wait for this file to be written."
    )
    parser.add_argument("--wait_timeout", type=float, default=15.0, help="Seconds to wait for --wait_for_file.")
    parser.add_argument("--raw", action="store_true", help="Print the server's JSON reply as is.")
    args = parser.parse_args(argv)

    if args.code is not None:
        code = args.code
    elif args.file is not None:
        code = args.file.read_text()
    else:
        code = sys.stdin.read()

    try:
        reply = send_request(args.host, args.port, code, args.timeout)
    except ConnectionRefusedError:
        print(
            f"No server on {args.host}:{args.port}. Is the simulation still starting, or was it launched with "
            "--kit_args '--enable isaacsim.code_editor.python_server ...' and this port?",
            file=sys.stderr,
        )
        return 2
    except TimeoutError:
        print(f"No reply from {args.host}:{args.port} within {args.timeout} s. {NO_REPLY_HINT}", file=sys.stderr)
        return 2

    if args.raw:
        print(json.dumps(reply))
    else:
        if reply.get("output"):
            print(reply["output"])
        if reply.get("result") is not None:
            print(json.dumps(reply["result"]))
    if reply.get("status") != "ok":
        print(f"{reply.get('ename')}: {reply.get('evalue')}", file=sys.stderr)
        print("\n".join(reply.get("traceback") or []), file=sys.stderr)
        return 1

    if args.wait_for_file is not None:
        if not wait_for_file(args.wait_for_file, args.wait_timeout):
            print(f"{args.wait_for_file} was not written within {args.wait_timeout} s.", file=sys.stderr)
            return 3
        print(args.wait_for_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
