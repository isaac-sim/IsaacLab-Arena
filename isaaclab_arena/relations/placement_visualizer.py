# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rerun debug view of build-time placement validation, sim-free (no SimApp).

Rerun's viewer is a separate process fed by the logging SDK, so nothing here touches Isaac Sim -- the
window comes up while layouts are being solved, before any simulation exists.

Turn it on from an env graph YAML (worked example:
``isaaclab_arena/tests/test_data/placement_debug_view_env_graph.yaml``)::

    placement_validators:
      debug_visualize: true                          # spawn a viewer window; needs a reachable display
      debug_visualize_rrd_path: /tmp/placement.rrd   # and/or record, for headless runs

or in Python with ``ObjectPlacerParams(debug_visualize=True)``. Either field alone enables the view.
Shipped envs leave it off, since it spawns a window on every build; enable it while debugging a scene
whose layouts look wrong, then take it back out.

Every candidate layout is one frame of the ``candidate`` timeline, so scrubbing it shows what was
solved and which checks rejected it. Checks that know more about a candidate than its boxes add their
own layer under ``world/robot`` (see the cuRobo reachability check).

The spawned window belongs to the run: it comes up during placement, stays up for the rest of the
run, and dies with the process that spawned it. Record to an ``.rrd`` to inspect layouts afterwards.
"""

from __future__ import annotations

import math
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

CANDIDATE_TIMELINE = "candidate"
"""Rerun timeline whose sequence index is the candidate layout number."""

LAYOUT_ENTITY = "world/layout"
"""Entity path of the candidate's object boxes."""

ROBOT_ENTITY = "world/robot"
"""Entity path reserved for check-specific robot layers; cleared per candidate so a check that skips a
candidate does not leave its previous frame's geometry on screen."""

ANCHOR_COLOR = (140, 140, 150)
"""Color of the layout's anchors, which are fixed and only act as obstacles."""

MOVABLE_COLOR = (70, 130, 220)
"""Color of the objects placement actually solves for."""

VIEWER_HOST = "127.0.0.1"
"""Interface the spawned viewer is reached on; it always runs alongside the process that logs to it."""

VIEWER_PORT = 9876
"""Port the spawned viewer serves on; Rerun's default, so ``rerun --connect`` finds it unprompted."""

VIEWER_SHUTDOWN_TIMEOUT_S = 10.0
"""How long an explicit close() waits for the viewer window to go away before giving up on it."""

VIEWER_STARTUP_TIMEOUT_S = 20.0
"""How long spawning waits for the viewer window to start serving before letting placement run on."""

VIEWER_PROBE_TIMEOUT_S = 0.2
"""How long one connection attempt to the viewer port may take before it counts as unanswered."""

VIEWER_PROBE_INTERVAL_S = 0.1
"""How long to wait between connection attempts while the viewer window starts."""

_ACTIVE_VISUALIZER: PlacementRerunVisualizer | None = None
"""The process's live view. Placement builds several placers (pool, per-reset solves) that would
otherwise each reset Rerun's global recording and fight over the same viewer port and ``.rrd``."""


def get_or_create_placement_visualizer(params: ObjectPlacerParams) -> PlacementRerunVisualizer | None:
    """Return the process's Rerun view of placement validation, or None when the params ask for none.

    Args:
        params: Placement parameters carrying the ``debug_visualize`` / ``debug_visualize_rrd_path`` fields.
    """
    global _ACTIVE_VISUALIZER
    if not params.debug_visualize and params.debug_visualize_rrd_path is None:
        return None
    if _ACTIVE_VISUALIZER is None:
        _ACTIVE_VISUALIZER = PlacementRerunVisualizer(
            spawn=params.debug_visualize, rrd_path=params.debug_visualize_rrd_path
        )
    return _ACTIVE_VISUALIZER


def find_rerun_viewer_executable() -> str | None:
    """Return the path of the Rerun viewer binary shipped with ``rerun-sdk``, or None if absent.

    Isaac Sim's Python does not put the packaged ``rerun_cli`` directory on PATH, so ``rr.spawn()``
    fails to find the viewer unless it is passed explicitly.
    """
    import rerun as rr

    executable = Path(rr.__file__).parents[1] / "rerun_cli" / "rerun"
    return str(executable) if executable.is_file() else None


def spawn_viewer_process() -> tuple[subprocess.Popen, Any]:
    """Spawn a viewer window that dies with this process; return it and the sink that streams to it.

    ``setpriv --pdeathsig`` rather than ``rerun.spawn()``, which detaches the viewer and drops its
    pid: the kernel then closes the window even on the hard ``os._exit`` Isaac Sim shuts down with.
    """
    import rerun as rr

    executable = find_rerun_viewer_executable()
    assert executable is not None, "rerun-sdk ships no viewer binary here; record to an .rrd instead."
    if _viewer_port_answers():
        print(
            f"WARNING: something already serves on port {VIEWER_PORT}; this run's candidate layouts will "
            "stream into that window rather than a new one."
        )
    viewer_process = subprocess.Popen([
        "setpriv",
        "--pdeathsig",
        "TERM",
        "--",
        executable,
        f"--port={VIEWER_PORT}",
        "--memory-limit=75%",
        "--server-memory-limit=1GiB",
        # Wait for this run's recording instead of opening on the welcome screen.
        "--expect-data-soon",
    ])
    _wait_until_viewer_serves(viewer_process)
    return viewer_process, rr.GrpcSink(url=f"rerun+http://{VIEWER_HOST}:{VIEWER_PORT}/proxy")


def _viewer_port_answers() -> bool:
    """Whether anything at all is serving on the viewer port -- not necessarily our own viewer."""
    with socket.socket() as probe:
        probe.settimeout(VIEWER_PROBE_TIMEOUT_S)
        return probe.connect_ex((VIEWER_HOST, VIEWER_PORT)) == 0


def _wait_until_viewer_serves(viewer_process: subprocess.Popen) -> None:
    """Block until the spawned viewer answers on its port, so the first candidates are not lost.

    Never fatal -- a view that fails to come up does not stop the run it was only meant to explain.
    """
    deadline = time.monotonic() + VIEWER_STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if _viewer_port_answers():
            # An answering port is not proof it is ours: a viewer that lost the race for it exits
            # right after, leaving this run logging into somebody else's window.
            if viewer_process.poll() is None:
                return
            print(f"WARNING: the Rerun viewer exited; layouts will stream to whatever else holds port {VIEWER_PORT}.")
            return
        if viewer_process.poll() is not None:
            print("WARNING: the Rerun viewer exited while starting; placement will not be visualized.")
            return
        time.sleep(VIEWER_PROBE_INTERVAL_S)
    print(f"WARNING: the Rerun viewer did not serve within {VIEWER_STARTUP_TIMEOUT_S:.0f}s; layouts may be missing.")


def summarize_candidate_verdict(
    candidate_index: int, verdicts_by_check: dict[str, bool], required_checks: set[str] | None
) -> tuple[str, bool]:
    """Describe how placement judged one candidate, as ``(message, accepted)``.

    Acceptance follows the placer: only required checks gate a layout, so a candidate that failed
    nothing else is accepted and the failure is reported as advisory rather than as a rejection.

    Args:
        candidate_index: Timeline index of the candidate, used in the message.
        verdicts_by_check: Verdict per check that ran on this candidate.
        required_checks: Checks that gate acceptance; None means every check that ran gates it.
    """
    failed = [check for check, passed in verdicts_by_check.items() if not passed]
    blocking = [check for check in failed if required_checks is None or check in required_checks]
    advisory = [check for check in failed if check not in blocking]
    if blocking:
        return f"candidate {candidate_index}: rejected (failed: {', '.join(blocking)})", False
    if advisory:
        return f"candidate {candidate_index}: accepted (failed but not required: {', '.join(advisory)})", True
    return f"candidate {candidate_index}: accepted", True


class PlacementRerunVisualizer:
    """Streams every validated candidate layout to Rerun, one frame per candidate."""

    def __init__(self, app_id: str = "arena_placement", spawn: bool = True, rrd_path: str | None = None) -> None:
        """Start the recording and, unless recording headlessly, spawn a viewer window.

        Args:
            app_id: Rerun application id, shown in the viewer title.
            spawn: Whether to spawn a local viewer window and stream to it.
            rrd_path: Optional path to also record the stream to, for replay on another machine.
        """
        import rerun as rr

        rr.init(app_id, spawn=False)
        sinks: list = []
        self._viewer_process: subprocess.Popen | None = None
        if spawn:
            self._viewer_process, viewer_sink = spawn_viewer_process()
            sinks.append(viewer_sink)
        if rrd_path is not None:
            Path(rrd_path).parent.mkdir(parents=True, exist_ok=True)
            sinks.append(rr.FileSink(rrd_path))
        assert sinks, "PlacementRerunVisualizer needs a viewer to spawn or an .rrd path to record to."
        rr.set_sinks(*sinks)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._next_candidate_index = 0
        self._active_candidate_indices: list[int] = []

    def __deepcopy__(self, memory_map: dict[int, object]) -> PlacementRerunVisualizer:
        """Return the live view for ``copy.deepcopy`` instead of duplicating it.

        Isaac Lab's configclass deep-copies the placement event params that carry the pool, and a
        duplicated view would keep its own candidate counter and overwrite frames this one already drew.

        Args:
            memory_map: ``copy.deepcopy``'s ``id(original) -> copy`` cache.
        """
        memory_map[id(self)] = self
        return self

    @property
    def num_logged_candidates(self) -> int:
        """How many candidate layouts have been given a frame so far."""
        return self._next_candidate_index

    def next_batch_indices(self, num_candidates: int) -> list[int]:
        """Reserve and return one timeline index per candidate of the batch about to be validated.

        Indices keep counting across batches so a pool that refills several times does not overwrite
        its earlier frames.
        """
        start = self._next_candidate_index
        self._next_candidate_index += num_candidates
        return list(range(start, self._next_candidate_index))

    def set_active_candidates(self, candidate_indices: list[int]) -> None:
        """Declare which candidates the validator about to run will see, in the order it sees them.

        Expensive checks only run on the candidates that passed the cheap ones, so their batch position
        is not the candidate number; this is what lets them log against the right frame.
        """
        self._active_candidate_indices = list(candidate_indices)

    def candidate_index_for_slot(self, slot: int) -> int:
        """Timeline index of the ``slot``-th candidate in the batch the running validator was given."""
        return self._active_candidate_indices[slot]

    def set_time(self, candidate_index: int) -> None:
        """Point the recording at one candidate's frame, so subsequent logs land on it."""
        import rerun as rr

        rr.set_time(CANDIDATE_TIMELINE, sequence=candidate_index)

    def log_layout(
        self,
        candidate_index: int,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        orientations: dict[PlaceableAsset, float],
        bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        anchors: set[PlaceableAsset],
    ) -> None:
        """Log one candidate's solved layout as boxes in the world frame.

        Args:
            candidate_index: Timeline index to log against.
            positions: Solved (x, y, z) per object.
            orientations: Absolute world Z-yaw per object; objects without one are drawn unrotated.
            bboxes: Per-object local bounding box.
            anchors: The layout's anchor objects, drawn in the anchor color.
        """
        import rerun as rr

        self.set_time(candidate_index)
        # A check that skips this candidate must not leave its previous candidate's robot on screen.
        rr.log(ROBOT_ENTITY, rr.Clear(recursive=True))

        objects = list(positions)
        centers, half_sizes, quaternions = [], [], []
        for obj in objects:
            yaw = orientations.get(obj, 0.0)
            bbox = bboxes[obj]
            local_center = [float(v) for v in bbox.center[0].tolist()]
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            rotated_center = (
                cos_yaw * local_center[0] - sin_yaw * local_center[1],
                sin_yaw * local_center[0] + cos_yaw * local_center[1],
                local_center[2],
            )
            position = positions[obj]
            centers.append([position[i] + rotated_center[i] for i in range(3)])
            half_sizes.append([0.5 * float(v) for v in bbox.size[0].tolist()])
            quaternions.append(rr.Quaternion(xyzw=[0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]))

        rr.log(
            LAYOUT_ENTITY,
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                quaternions=quaternions,
                colors=[ANCHOR_COLOR if obj in anchors else MOVABLE_COLOR for obj in objects],
                labels=[obj.name for obj in objects],
                fill_mode=rr.components.FillMode.MajorWireframe,
            ),
        )

    def log_verdicts(
        self, candidate_index: int, verdicts_by_check: dict[str, bool], required_checks: set[str] | None
    ) -> None:
        """Log which checks accepted one candidate, as a text line and an accepted/rejected marker.

        Args:
            candidate_index: Timeline index to log against.
            verdicts_by_check: Verdict per check that ran on this candidate; one that skipped it is absent.
            required_checks: Checks that gate acceptance; None means every check that ran gates it.
        """
        import rerun as rr

        self.set_time(candidate_index)
        message, accepted = summarize_candidate_verdict(candidate_index, verdicts_by_check, required_checks)
        rr.log(
            f"{LAYOUT_ENTITY}/verdict",
            rr.TextLog(message, level=rr.TextLogLevel.INFO if accepted else rr.TextLogLevel.WARN),
        )
        for check, passed in verdicts_by_check.items():
            rr.log(f"checks/{check}", rr.Scalars(float(passed)))

    def close(self) -> None:
        """Flush pending data and shut down the viewer window this run spawned. Idempotent.

        Only needed to close the window early -- a run that just exits leaves it to the viewer's
        parent-death signal.
        """
        import rerun as rr

        # None once Rerun's own shutdown hook has torn the recording down ahead of this call.
        recording = rr.get_global_data_recording()
        if recording is not None:
            recording.flush()
        viewer_process, self._viewer_process = self._viewer_process, None
        if viewer_process is None:
            return
        viewer_process.terminate()
        try:
            viewer_process.wait(timeout=VIEWER_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # A window wedged past SIGTERM would otherwise keep the port and outlive the run.
            viewer_process.kill()
            viewer_process.wait()
