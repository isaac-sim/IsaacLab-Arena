# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import subprocess
import sys
import traceback
from collections.abc import Callable

from isaaclab.app import AppLauncher
from isaacsim import SimulationApp

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.tests.conftest import PYTEST_SESSION
from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher, teardown_simulation_app

# NOTE(alexmillane): Isaac Sim makes testing complicated. During shutdown Isaac Sim will
# terminate the surrounding pytest process with exit code 0, regardless
# of whether the tests passed or failed.
# To work around this, we track the failure state of the tests in two ways:
# 1. We stash the pytest session object and set a flag when a test fails.
# 2. We set a flag when a test fails.
# These flags are checked in prior to closing the simulation app in _close_persistent(),
# and we manually exit the process with the exit code 1 if tests have failed.

_PERSISTENT_SIM_APP_LAUNCHER: AppLauncher | None = None
_PERSISTENT_INIT_ARGS = None  # store (headless, enable_cameras) used at first init
_AT_LEAST_ONE_TEST_FAILED = False


_SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT", "600"))


def run_subprocess(
    cmd,
    env=None,
    timeout_sec: int | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess | None:
    """Run a command in a subprocess with timeout.

    The child is launched with ``start_new_session=True`` so it lives in its
    own process group.  The child-side ``SimulationAppContext`` uses this to
    SIGTERM its entire group before ``os._exit()``, preventing orphaned Kit
    children (shader compiler, GPU workers, …) from holding GPU resources and
    blocking the next subprocess.

    Args:
        cmd: Command to run (list of strings).
        env: Optional environment dict.  Defaults to inheriting the parent env.
        timeout_sec: Per-subprocess wall-clock timeout in seconds.
            Defaults to ``_SUBPROCESS_TIMEOUT_SEC`` (env ``ISAACLAB_ARENA_SUBPROCESS_TIMEOUT``, fallback 600).
        capture_output: If True, capture stdout/stderr and return a
            ``CompletedProcess``.  When False (default) output streams to
            the parent process and the function returns None on success.

    Returns:
        ``CompletedProcess`` when *capture_output* is True, else None.
    """
    if timeout_sec is None:
        timeout_sec = _SUBPROCESS_TIMEOUT_SEC

    print(f"Running command (timeout={timeout_sec}s): {cmd}")
    global _AT_LEAST_ONE_TEST_FAILED

    if env is None:
        env = os.environ.copy()
    env["ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE"] = "1"

    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=timeout_sec,
            capture_output=capture_output,
            text=capture_output,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"\n[isaaclab-arena] Subprocess timed out after {timeout_sec}s\n")
        _AT_LEAST_ONE_TEST_FAILED = True
        raise subprocess.SubprocessError(f"Subprocess timed out after {timeout_sec}s: {cmd}")

    print(f"Command completed with return code: {result.returncode}")
    if result.returncode != 0:
        sys.stderr.write(f"Command failed with return code {result.returncode}\n")
        if capture_output and result.stderr:
            sys.stderr.write(result.stderr)
        _AT_LEAST_ONE_TEST_FAILED = True
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    if capture_output:
        return result
    return None


class _IsolatedArgv:
    """Temporarily replace sys.argv so Kit doesn't see pytest flags like '-m'."""

    def __init__(self, argv=None):
        # Keep program name; drop the rest (or use provided list)
        self._new = [sys.argv[0]] + (argv or [])
        self._old = None

    def __enter__(self):
        self._old = sys.argv[:]
        sys.argv = self._new

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self._old


def _close_persistent():
    global _PERSISTENT_SIM_APP_LAUNCHER
    global _AT_LEAST_ONE_TEST_FAILED
    if _PERSISTENT_SIM_APP_LAUNCHER is not None:
        tests_failed = PYTEST_SESSION.tests_failed or _AT_LEAST_ONE_TEST_FAILED
        print(f"Closing persistent simulation app. Tests failed: {tests_failed}")
        if tests_failed:
            # If any test failed, exit the process with exit code 1
            # to prevent Isaac Sim from terminating the pytest process with exit code 0.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        else:
            _PERSISTENT_SIM_APP_LAUNCHER.app.close()


def get_persistent_simulation_app(headless: bool, enable_cameras: bool = False) -> SimulationApp:
    """Create once, reuse forever (until process exit)."""
    global _PERSISTENT_SIM_APP_LAUNCHER, _PERSISTENT_INIT_ARGS
    # Create a new simulation app if it doesn't exist
    if _PERSISTENT_SIM_APP_LAUNCHER is None:
        parser = get_isaaclab_arena_cli_parser()
        simulation_app_args = parser.parse_args([])
        simulation_app_args.headless = headless
        simulation_app_args.enable_cameras = enable_cameras
        if not headless:
            simulation_app_args.visualizer = ["kit"]
        with _IsolatedArgv([]):

            app_launcher = get_app_launcher(simulation_app_args)

        _PERSISTENT_SIM_APP_LAUNCHER = app_launcher
        _PERSISTENT_INIT_ARGS = (headless, enable_cameras)
        atexit.register(_close_persistent)
    else:
        # sanity-check mismatched flags after first init
        first_headless, first_enable_cameras = _PERSISTENT_INIT_ARGS
        if (headless != first_headless) or (enable_cameras != first_enable_cameras):
            print(
                "[isaaclab-arena] Warning: persistent SimulationApp already initialized with "
                f"headless={first_headless}, enable_cameras={first_enable_cameras}. "
                "Ignoring new values."
            )
    return _PERSISTENT_SIM_APP_LAUNCHER.app


def run_simulation_app_function(
    function: Callable[..., bool],
    headless: bool = True,
    enable_cameras: bool = False,
    **kwargs,
) -> bool:
    """Run a simulation app in a separate process.

    This is sometimes required to prevent simulation app shutdown interrupting pytest.

    Args:
        function: The function to run in the simulation app.
            - The function should take a SimulationAppContext instance as its first argument,
            and then a variable number of additional arguments.
            - The function should return a boolean indicating whether the test passed.
        *args: The arguments to pass to the function (after the SimulationAppContext instance).

    Returns:
        The boolean result of the function.
    """
    # Get a persistent simulation app
    global _AT_LEAST_ONE_TEST_FAILED
    try:
        simulation_app = get_persistent_simulation_app(headless=headless, enable_cameras=enable_cameras)
        test_result = bool(function(simulation_app, **kwargs))
        if not test_result:
            _AT_LEAST_ONE_TEST_FAILED = True
        return test_result
    except Exception as e:
        print(f"Exception occurred while running the function (persistent mode): {e}")
        traceback.print_exc()
        return False
    finally:
        # **Always** clean up the SimulationContext/timeline between tests
        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
