# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import traceback

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
