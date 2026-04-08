# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import traceback
from contextlib import nullcontext, suppress

import omni.kit.app
from isaaclab.app import AppLauncher


def get_isaac_sim_version() -> str:
    """Get the version of Isaac Sim."""
    return omni.kit.app.get_app().get_app_version()


def get_app_launcher(args: argparse.Namespace) -> AppLauncher:
    """Get an app launcher."""
    app_launcher = AppLauncher(args)
    return app_launcher


def teardown_simulation_app(suppress_exceptions: bool = False, make_new_stage: bool = True) -> None:
    """
    Tear down the SimulationApp and start a fresh USD stage preparing for the next content.
    Useful for loading new content into the SimulationApp without restarting the app.

    Args:
        suppress_exceptions: Whether to suppress exceptions. If True, the exception will be caught and the execution will continue. If False, the exception will be propagated.
        make_new_stage: Whether to make a new USD stage. If True, a new USD stage will be created. If False, the current USD stage will be used.
    """
    if suppress_exceptions:
        # silently caught exceptions and continue the execution.
        error_manager = suppress(Exception)
    else:
        # Do nothing and let the exception to be raised.
        error_manager = nullcontext()

    with error_manager:
        # Local import to avoid loading Isaac/Kit unless needed.
        from isaaclab.sim import SimulationContext

        sim = None
        with error_manager:
            sim = SimulationContext.instance()

        # Stop the simulation app
        if sim is not None:
            with error_manager:
                # Some versions gate shutdown on this flag.
                sim._disable_app_control_on_stop_handle = True  # noqa: SLF001 (intentional private attr)
            with error_manager:
                sim.stop()
            with error_manager:
                sim.clear_instance()

    # Stop the timeline
    with error_manager:
        import omni.timeline

        with error_manager:
            omni.timeline.get_timeline_interface().stop()

    # Finally, start a fresh USD stage for the next test
    if make_new_stage:
        with error_manager:
            import omni.usd

            omni.usd.get_context().new_stage()


def reapply_viewer_cfg(env) -> None:
    """Re-apply ViewerCfg camera position after visualizers are initialized.

    ViewportCameraController calls sim.set_camera_view() during __init__, but visualizers
    (e.g. KitVisualizer) are not yet initialized at that point and silently ignore the call.
    After gym.make() returns the visualizers are ready, so we call update_view_location()
    again to apply the configured eye/lookat position.
    """
    unwrapped = env.unwrapped
    vcc = getattr(unwrapped, "viewport_camera_controller", None)
    if vcc is not None:
        vcc.update_view_location()


class SimulationAppContext:
    """Context manager for launching and closing a simulation app."""

    def __init__(self, args: argparse.Namespace):
        """
        Args:
            args (argparse.Namespace): The arguments to the simulation app.
        """
        self.args = args
        self.app_launcher = None

    def is_running(self) -> bool:
        return self.app_launcher.app.is_running()

    def is_exiting(self) -> bool:
        return self.app_launcher.app.is_exiting()

    def __enter__(self):
        self.app_launcher = get_app_launcher(self.args)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing simulation app")
        if exc_type is not None:
            print(f"Exception caught in SimulationAppContext: {exc_type.__name__}: {exc_val}")
            print("Traceback:")
            traceback.print_exception(exc_type, exc_val, exc_tb)
            print("Killing the process without cleaning up")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

        # When launched as a test subprocess, skip app.close() which can hang
        # indefinitely in Kit's shutdown path.  The parent process owns the
        # lifetime via process-group kill (see run_subprocess).
        if os.environ.get("ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE") == "1":
            print("Force-exiting subprocess (ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1)")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        # Normal interactive / non-test path: attempt a clean Kit shutdown.
        # app.close() may terminate the process with exit code 0 regardless of
        # errors — see the error branch above for the workaround.
        self.app_launcher.app.close()
