# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import os
import sys
import traceback
from contextlib import ExitStack, nullcontext, suppress
from types import SimpleNamespace

from isaaclab.app import AppLauncher, launch_simulation, validate_launch

from isaaclab_arena.utils.camera_renderer import CameraRenderer
from isaaclab_arena.utils.physics_backend import PhysicsBackend


def get_isaac_sim_version() -> str:
    """Get the version of Isaac Sim."""
    import omni.kit.app

    return omni.kit.app.get_app().get_app_version()


STARTUP_COMPLETE_MARKER = "[isaaclab-arena] AppLauncher initialization complete"


def get_app_launcher(args: argparse.Namespace) -> AppLauncher:
    """Get an app launcher."""
    import time

    t0 = time.monotonic()
    # AppLauncher consumes recognized entries from its input mapping. Pass a copy so
    # Arena can continue using its parsed arguments after the application launches.
    app_launcher = AppLauncher(vars(args).copy())
    elapsed = time.monotonic() - t0
    sys.__stderr__.write(f"{STARTUP_COMPLETE_MARKER} ({elapsed:.1f}s)\n")
    sys.__stderr__.flush()
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
            from isaaclab.sim.utils import close_stage, get_current_stage

            # SimulationContext cleanup clears Lab's stage; USD-only work does not.
            # Close only a lingering Lab stage to avoid closing Kit twice.
            if get_current_stage() is not None:
                close_stage()
            omni.usd.get_context().new_stage()


def collect_garbage_and_clear_cuda_cache() -> None:
    """Run GC and release cached CUDA allocations after a sim env is torn down."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _kill_child_processes() -> None:
    """SIGKILL all direct child processes of the current process via /proc."""
    import signal

    my_pid = os.getpid()
    with suppress(FileNotFoundError, PermissionError):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            if int(line.split()[1]) == my_pid:
                                os.kill(int(entry.name), signal.SIGKILL)
                            break
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue


def _newton_launch_cfg(args: argparse.Namespace) -> SimpleNamespace:
    """Return a stand-in for the environment config of a Newton run.

    Isaac Lab's launch reads the physics and the camera renderer of the config to decide whether to start Kit. Arena
    builds the environment only after the launch, so the arguments stand in for it.
    """
    from isaaclab_newton.physics import NewtonCfg

    cameras = CameraRenderer(args.camera_renderer).renderer_cfg() if args.enable_cameras else None
    return SimpleNamespace(physics=NewtonCfg(), cameras=cameras)


def _needs_kit(args: argparse.Namespace, require_kit: bool) -> bool:
    """Return whether a run with these arguments needs Kit, as described in ``SimulationAppContext``."""
    # Not every entry point has --presets.
    if require_kit or getattr(args, "presets", None) != PhysicsBackend.NEWTON:
        return True
    launch_cfg = _newton_launch_cfg(args)
    require_kit = (
        (args.enable_cameras and launch_cfg.cameras is None) or args.xr or args.kit_args or os.environ.get("XR") == "1"
    )
    try:
        _, kit_sources = validate_launch(launch_cfg, vars(args) | {"require_kit": require_kit})
    except ValueError:
        # Isaac Lab rejects some combinations without Kit, such as newton_rtx with a Kit visualizer. AppLauncher then
        # drops newton_rtx and starts Kit.
        return True
    return bool(kit_sources)


class SimulationAppContext:
    """Context manager for launching and closing a simulation app.

    Kit starts unless the run selects Newton physics (``--presets newton``) and Isaac Lab finds nothing that needs Kit,
    such as Isaac RTX cameras, XR or ``--viz kit``.
    """

    def __init__(self, args: argparse.Namespace, require_kit: bool = False):
        """
        Args:
            args (argparse.Namespace): The arguments to the simulation app.
            require_kit: Start Kit even if the arguments allow a launch without it.
        """
        self.args = args
        self.require_kit = require_kit
        self.app_launcher = None
        self._exit_stack = ExitStack()

    def is_running(self) -> bool:
        if self.app_launcher is None:
            from isaaclab.sim import SimulationContext

            # Without Kit, a run goes on until the viewers of its simulation close; before it exists, nothing stopped.
            simulation = SimulationContext.instance()
            return simulation is None or simulation.is_running()
        return self.app_launcher.app.is_running()

    def is_exiting(self) -> bool:
        return self.app_launcher is not None and self.app_launcher.app.is_exiting()

    def __enter__(self):
        if _needs_kit(self.args, self.require_kit):
            self.app_launcher = get_app_launcher(self.args)
            self._exit_stack.callback(self.app_launcher.app.close)
        else:
            self._exit_stack.enter_context(launch_simulation(_newton_launch_cfg(self.args), vars(self.args).copy()))
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
        # indefinitely in Kit's shutdown path.
        if os.environ.get("ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE") == "1":
            print("Force-exiting subprocess (ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1)")
            sys.stdout.flush()
            sys.stderr.flush()
            # SIGKILL orphaned Kit children (shader compiler, GPU workers, …)
            # so they don't hold GPU resources and block the next test subprocess.
            # We target each child individually via /proc to avoid signalling
            # ourselves (Kit installs a C-level SIGTERM handler that overrides
            # Python's SIG_IGN, so os.killpg is not safe here).
            _kill_child_processes()
            os._exit(0)

        # Normal interactive / non-test path: attempt a clean shutdown.
        # app.close() may terminate the process with exit code 0 regardless of
        # errors — see the error branch above for the workaround.
        self._exit_stack.close()
