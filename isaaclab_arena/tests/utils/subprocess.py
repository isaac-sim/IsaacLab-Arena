# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_PERSISTENT_SIM_APP: SimulationAppContext | None = None
_PERSISTENT_INIT_ARGS = None  # store (headless, enable_cameras) used at first init


def run_subprocess(cmd, env=None):
    print(f"Running command: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            # Don't capture output, let it flow through in real-time
            capture_output=False,
            text=True,
            # Explicitly set stdout and stderr to None to use parent process's pipes
            stdout=None,
            stderr=None,
        )
        print(f"Command completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Command failed with return code {e.returncode}: {e}\n")
        raise


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


def safe_teardown(make_new_stage: bool = True) -> None:
    """
    Best-effort reset so the persistent SimulationApp can accept the next test.
    Runs even if a test fails or raises.
    """
    with suppress(Exception):
        # Local import to avoid loading Isaac/Kit unless needed.
        from isaaclab.sim import SimulationContext

        sim = None
        with suppress(Exception):
            sim = SimulationContext.instance()

        # Stop the simulation app
        if sim is not None:
            with suppress(Exception):
                # Some versions gate shutdown on this flag.
                sim._disable_app_control_on_stop_handle = True  # noqa: SLF001 (intentional private attr)
            with suppress(Exception):
                sim.stop()
            with suppress(Exception):
                sim.clear_instance()

    # Stop the timeline
    with suppress(Exception):
        import omni.timeline

        with suppress(Exception):
            omni.timeline.get_timeline_interface().stop()

    # Finally, start a fresh USD stage for the next test
    if make_new_stage:
        with suppress(Exception):
            import omni.usd

            omni.usd.get_context().new_stage()


def _close_persistent():
    global _PERSISTENT_SIM_APP
    if _PERSISTENT_SIM_APP is not None:
        try:
            # mirror context-manager exit to shut down cleanly at process end
            _PERSISTENT_SIM_APP.__exit__(None, None, None)
        finally:
            _PERSISTENT_SIM_APP = None


def get_persistent_simulation_app(headless: bool, enable_cameras: bool = False) -> SimulationAppContext:
    """Create once, reuse forever (until process exit)."""
    global _PERSISTENT_SIM_APP, _PERSISTENT_INIT_ARGS
    # Create a new simulation app if it doesn't exist
    if _PERSISTENT_SIM_APP is None:
        parser = get_isaaclab_arena_cli_parser()
        simulation_app_args = parser.parse_args([])
        simulation_app_args.headless = headless
        simulation_app_args.enable_cameras = enable_cameras
        with _IsolatedArgv([]):
            app = SimulationAppContext(simulation_app_args)
            # Manually "enter" the context manager so we can keep it open
            app.__enter__()
        _PERSISTENT_SIM_APP = app
        _PERSISTENT_INIT_ARGS = (headless, enable_cameras)
        atexit.register(_close_persistent)
    else:
        # sanity-check mismatched flags after first init
        first_headless, first_enable_cameras = _PERSISTENT_INIT_ARGS
        if (headless != first_headless) or (enable_cameras != first_enable_cameras):
            print(
                "[isaac-arena] Warning: persistent SimulationApp already initialized with "
                f"headless={first_headless}, enable_cameras={first_enable_cameras}. "
                "Ignoring new values."
            )
    return _PERSISTENT_SIM_APP


def run_simulation_app_function(
    function: Callable[..., bool], headless: bool = True, enable_cameras: bool = False, **kwargs
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
    try:
        simulation_app = get_persistent_simulation_app(headless=headless, enable_cameras=enable_cameras)
        return bool(function(simulation_app, **kwargs))
    except Exception as e:
        print(f"Exception occurred while running the function (persistent mode): {e}")
        return False
    finally:
        # **Always** clean up the SimulationContext/timeline between tests
        safe_teardown()
