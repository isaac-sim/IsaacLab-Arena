# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for run_in_sim.py against a fake Python server; no simulation is started."""

import io
import json
import socket
import socketserver
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from isaaclab_arena.agentic_tools.sim_inspection.run_in_sim import main


class _FakePythonServerHandler(socketserver.BaseRequestHandler):
    """Read one request until the client half-closes, then send the server's canned reply."""

    def handle(self):
        chunks = []
        while True:
            chunk = self.request.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        self.server.received_requests.append(b"".join(chunks).decode())
        if self.server.reply is None:
            self.server.release_silent_handlers.wait(timeout=10)
            return
        self.request.sendall(json.dumps(self.server.reply).encode())


class FakePythonServer(socketserver.ThreadingTCPServer):
    """Stand-in for isaacsim.code_editor.python_server; reply=None means never answer."""

    daemon_threads = True

    def __init__(self, reply):
        super().__init__(("127.0.0.1", 0), _FakePythonServerHandler)
        self.reply = reply
        self.received_requests = []
        self.release_silent_handlers = threading.Event()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self):
        return self.server_address[1]

    def stop(self):
        self.release_silent_handlers.set()
        self.shutdown()
        self.server_close()


def run_client(argv, stdin_text=""):
    """Run the client's main() and capture its exit code, stdout and stderr."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr), patch("sys.stdin", io.StringIO(stdin_text)):
        exit_code = main(argv)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def unused_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RunInSimTest(unittest.TestCase):
    def start_server(self, reply):
        server = FakePythonServer(reply)
        self.addCleanup(server.stop)
        return server

    def test_sends_code_in_json_envelope(self):
        server = self.start_server({"status": "ok", "output": "hello from sim"})
        exit_code, stdout, _ = run_client(["--port", str(server.port), "--code", "print(1)"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(server.received_requests[0]), {"code": "print(1)", "timeout": 30.0})
        self.assertEqual(stdout.strip(), "hello from sim")

    def test_prints_result_as_json_line(self):
        server = self.start_server({"status": "ok", "output": "", "result": {"a": 1}})
        exit_code, stdout, _ = run_client(["--port", str(server.port), "--code", "{'a': 1}"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.strip()), {"a": 1})

    def test_none_result_prints_nothing(self):
        server = self.start_server({"status": "ok", "output": "", "result": None})
        exit_code, stdout, _ = run_client(["--port", str(server.port), "--code", "set_camera([1, 1, 1], [0, 0, 0])"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "")

    def test_error_reply_exits_1_with_traceback(self):
        server = self.start_server({
            "status": "error",
            "output": "",
            "ename": "NameError",
            "evalue": "name 'capture_viewer' is not defined",
            "traceback": ["Traceback (most recent call last):\n  line 1\nNameError"],
        })
        exit_code, _, stderr = run_client(["--port", str(server.port), "--code", "capture_viewer('/tmp/x.png')"])
        self.assertEqual(exit_code, 1)
        self.assertTrue("NameError: name 'capture_viewer' is not defined" in stderr, stderr)
        self.assertTrue("Traceback (most recent call last)" in stderr, stderr)

    def test_connection_refused_exits_2(self):
        port = unused_port()
        exit_code, _, stderr = run_client(["--port", str(port), "--code", "print(1)"])
        self.assertEqual(exit_code, 2)
        self.assertTrue("no server" in stderr.lower(), stderr)
        self.assertTrue(str(port) in stderr, stderr)

    def test_no_reply_times_out_with_launch_mode_hint(self):
        server = self.start_server(reply=None)
        started = time.monotonic()
        exit_code, _, stderr = run_client(["--port", str(server.port), "--code", "print(1)", "--timeout", "0.5"])
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(exit_code, 2)
        self.assertTrue("no reply" in stderr.lower(), stderr)
        self.assertTrue("updates Kit every step" in stderr, stderr)

    def test_reads_code_from_file_and_stdin(self):
        server = self.start_server({"status": "ok", "output": ""})
        with tempfile.TemporaryDirectory() as temp_dir:
            code_path = Path(temp_dir) / "request.py"
            code_path.write_text("x = 2\n")
            self.assertEqual(run_client(["--port", str(server.port), "--file", str(code_path)])[0], 0)
        self.assertEqual(run_client(["--port", str(server.port)], stdin_text="y = 3\n")[0], 0)
        sent_code = [json.loads(request)["code"] for request in server.received_requests]
        self.assertEqual(sent_code, ["x = 2\n", "y = 3\n"])

    def test_wait_for_file_returns_after_file_appears(self):
        server = self.start_server({"status": "ok", "output": ""})
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "view.png"
            writer = threading.Timer(0.3, image_path.write_bytes, args=(b"png bytes",))
            writer.start()
            self.addCleanup(writer.cancel)
            exit_code, stdout, _ = run_client(
                ["--port", str(server.port), "--code", "pass", "--wait_for_file", str(image_path)]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(str(image_path) in stdout, stdout)

    def test_wait_for_file_ignores_previous_capture(self):
        server = self.start_server({"status": "ok", "output": ""})
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "view.png"
            image_path.write_bytes(b"previous capture")
            writer = threading.Timer(0.5, image_path.write_bytes, args=(b"new capture",))
            writer.start()
            self.addCleanup(writer.cancel)
            exit_code, _, _ = run_client(
                ["--port", str(server.port), "--code", "pass", "--wait_for_file", str(image_path)]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(image_path.read_bytes(), b"new capture")

    def test_wait_for_file_times_out_exits_3(self):
        server = self.start_server({"status": "ok", "output": ""})
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "never.png"
            exit_code, _, stderr = run_client([
                "--port",
                str(server.port),
                "--code",
                "pass",
                "--wait_for_file",
                str(missing_path),
                "--wait_timeout",
                "0.5",
            ])
        self.assertEqual(exit_code, 3)
        self.assertTrue(str(missing_path) in stderr, stderr)


if __name__ == "__main__":
    unittest.main()
