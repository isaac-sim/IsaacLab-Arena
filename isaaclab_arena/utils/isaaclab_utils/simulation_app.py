# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    # NOTE(alexmillane, 2025.11.10): Import pinocchio before launching the app appears still to be required.
    # Monitor this and see if we can get rid of it.
    if hasattr(args, "enable_pinocchio") and args.enable_pinocchio:
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(args)
    if get_isaac_sim_version() != "5.1.0":
        print(f"WARNING: IsaacSim has been upgraded to {get_isaac_sim_version()}.")
        print("Please investigate if pinocchio import is still needed in: simulation_app.py")
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
        # app_launcher.close() will terminate the whole process with exit code 0, i.e. preventing errors from being seen by the caller. There are seemingly no ways around this.
        # As a workaround, we call os._exit(1) that terminates immediately. The downside is that any cleanup would be omitted
        if exc_type is None:
            self.app_launcher.app.close()
        else:
            print(f"Exception caught in SimulationAppContext: {exc_type.__name__}: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            print("Killing the process without cleaning up")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
