# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rerun debug view of build-time placement validation, sim-free (no SimApp).

Turn it on with ``ObjectPlacerParams.debug_visualize`` for a viewer window, or
``debug_visualize_output_path`` for an ``.rrd`` recording; either one alone is enough. From an env
graph YAML (worked example: ``isaaclab_arena/tests/test_data/placement_debug_view_env_graph.yaml``)::

    placement_validators:
      debug_visualize: true
      debug_visualize_output_path: /tmp/placement.rrd
"""

from __future__ import annotations

import math
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.relations.relations import get_anchor_objects

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

LAYOUT_TIMELINE = "layout"
"""Rerun timeline whose sequence index is the layout number."""

LAYOUT_ENTITY = "world/layout"
"""Entity path of the layout's object boxes."""

ROBOT_ENTITY = "world/robot"
"""Entity path for check-specific robot layers; cleared on every layout, so nothing carries over."""

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
"""The process's live view tied to a placer: the recording, the viewer port and the output path."""


def get_or_create_placement_visualizer(params: ObjectPlacerParams) -> PlacementRerunVisualizer | None:
    """Return the process's Rerun view of placement validation, or None when no one asks for it.

    Args:
        params: Placement parameters carrying the ``debug_visualize`` / ``debug_visualize_output_path`` fields.
    """
    # Global to avoid creating multiple visualizers for the same process.
    global _ACTIVE_VISUALIZER
    if not params.debug_visualize and params.debug_visualize_output_path is None:
        return None
    if _ACTIVE_VISUALIZER is None:
        _ACTIVE_VISUALIZER = PlacementRerunVisualizer(
            spawn=params.debug_visualize, output_path=params.debug_visualize_output_path
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
            f"WARNING: something already serves on port {VIEWER_PORT}; this run's layouts will "
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
    """Block until the spawned viewer answers on its port, so the first layouts are not lost.

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


def summarize_layout_verdict(
    layout_index_across_batch: int, verdicts_by_check: dict[str, bool], required_checks: set[str] | None
) -> tuple[str, bool]:
    """Describe how placement judged one layout, as ``(message, accepted)``.

    Args:
        layout_index_across_batch: Timeline index of the layout, used in the message.
        verdicts_by_check: Verdict per check that ran on this layout.
        required_checks: Checks that gate acceptance; None means every check that ran gates it.
    """
    failed = [check for check, passed in verdicts_by_check.items() if not passed]
    blocking = [check for check in failed if required_checks is None or check in required_checks]
    advisory = [check for check in failed if check not in blocking]
    layout = f"layout {layout_index_across_batch}"
    if blocking:
        return f"{layout}: rejected (failed: {', '.join(blocking)})", False
    if advisory:
        return f"{layout}: accepted (failed but not required: {', '.join(advisory)})", True
    return f"{layout}: accepted", True


class PlacementRerunVisualizer:
    """Draws every validated layout into a Rerun recording, one frame per layout."""

    def __init__(self, app_id: str = "arena_placement", spawn: bool = True, output_path: str | None = None) -> None:
        """Start the recording, streaming to a spawned viewer window and/or writing it to a file.

        Args:
            app_id: Rerun application id, shown in the viewer title.
            spawn: Whether to spawn a local viewer window and stream to it.
            output_path: Optional path to record the stream to.
        """
        import rerun as rr

        rr.init(app_id, spawn=False)
        sinks: list = []
        self._viewer_process: subprocess.Popen | None = None
        if spawn:
            self._viewer_process, viewer_sink = spawn_viewer_process()
            sinks.append(viewer_sink)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sinks.append(rr.FileSink(output_path))
        assert sinks, "PlacementRerunVisualizer needs a viewer to spawn or an output path to record to."
        rr.set_sinks(*sinks)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # Next unused timeline frame to attach to a layout, e.g. 5 after frame 0-4 are drawn
        self._next_layout_index_across_batch = 0

        # Across-batch index of each layout of the current batch, e.g. [3, 4] for a 2-layout batch after frames 0-2
        self._layout_indices_across_batch: list[int] = []

        # Across-batch index of the layout the next check runs on, e.g. [4] when only layout 1 of that batch passed
        self._active_layout_indices_across_batch: list[int] = []

    def __deepcopy__(self, memory_map: dict[int, object]) -> PlacementRerunVisualizer:
        """Return the live view for ``copy.deepcopy`` instead of duplicating it.

        Isaac Lab's configclass deep-copies the placement event params that carry the pool, and a
        duplicated view would keep its own layout counter and overwrite frames this one already drew.

        Args:
            memory_map: ``copy.deepcopy``'s ``id(original) -> copy`` cache.
        """
        memory_map[id(self)] = self
        return self

    @property
    def num_logged_layouts(self) -> int:
        """How many layouts have been given a frame so far."""
        return self._next_layout_index_across_batch

    def _reserve_layout_indices(self, num_layouts: int) -> list[int]:
        """Reserve and return one timeline index per layout of the batch about to be validated.

        Indices keep counting across batches so a pool that refills several times does not overwrite
        its earlier frames.

        Args:
            num_layouts: How many layouts the batch holds.
        """
        start = self._next_layout_index_across_batch
        self._next_layout_index_across_batch += num_layouts
        return list(range(start, self._next_layout_index_across_batch))

    def set_active_layouts(self, layout_indices_within_batch: list[int]) -> None:
        """Narrow the current batch to the layouts the next check runs on, in the order it sees them.

        Args:
            layout_indices_within_batch: Positions within the current batch, as the check received them.
        """
        self._active_layout_indices_across_batch = [
            self._layout_indices_across_batch[i] for i in layout_indices_within_batch
        ]

    def get_layout_index_across_batch(self, layout_index_within_batch: int) -> int:
        """Across-batch index of the layout the active subset holds at ``layout_index_within_batch``."""
        return self._active_layout_indices_across_batch[layout_index_within_batch]

    def set_time(self, layout_index_across_batch: int) -> None:
        """Point the recording at one layout's frame, so subsequent logs land on it."""
        import rerun as rr

        rr.set_time(LAYOUT_TIMELINE, sequence=layout_index_across_batch)

    def log_layout(
        self,
        layout_index_across_batch: int,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        orientations: dict[PlaceableAsset, float],
        bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        anchors: set[PlaceableAsset],
    ) -> None:
        """Draw one solved layout as boxes in the world frame.

        Args:
            layout_index_across_batch: Timeline index to log against.
            positions: Solved (x, y, z) per object.
            orientations: Absolute world Z-yaw per object; objects without one are drawn unrotated.
            bboxes: Per-object local bounding box.
            anchors: The layout's anchor objects, drawn in the anchor color.
        """
        import rerun as rr

        self.set_time(layout_index_across_batch)
        # A check that skips this layout must not leave its previous layout's robot on screen.
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

    def start_new_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
    ) -> None:
        """Give every solved layout of a new batch its own frame, and make that batch the current one.

        Every layout starts active, until a check narrows the batch with ``set_active_layouts``.

        Args:
            positions: Solved (x, y, z) per object, one dict per layout.
            orientations: Absolute world Z-yaw per object, one dict per layout.
            bboxes: Per-object local bounding box, one dict per layout.
        """
        self._layout_indices_across_batch = self._reserve_layout_indices(len(positions))
        for layout_index_within_batch, layout_index_across_batch in enumerate(self._layout_indices_across_batch):
            self.log_layout(
                layout_index_across_batch,
                positions[layout_index_within_batch],
                orientations[layout_index_within_batch],
                bboxes[layout_index_within_batch],
                anchors=set(get_anchor_objects(list(positions[layout_index_within_batch]))),
            )
        self._active_layout_indices_across_batch = list(self._layout_indices_across_batch)

    def log_layout_verdicts(
        self, layout_index_across_batch: int, verdicts_by_check: dict[str, bool], required_checks: set[str] | None
    ) -> None:
        """Log which checks accepted one layout, as a text line and an accepted/rejected marker.

        Args:
            layout_index_across_batch: Timeline index to log against.
            verdicts_by_check: Verdict per check that ran on this layout; one that skipped it is absent.
            required_checks: Checks that gate acceptance; None means every check that ran gates it.
        """
        import rerun as rr

        self.set_time(layout_index_across_batch)
        message, accepted = summarize_layout_verdict(layout_index_across_batch, verdicts_by_check, required_checks)
        rr.log(
            f"{LAYOUT_ENTITY}/verdict",
            rr.TextLog(message, level=rr.TextLogLevel.INFO if accepted else rr.TextLogLevel.WARN),
        )
        for check, passed in verdicts_by_check.items():
            rr.log(f"checks/{check}", rr.Scalars(float(passed)))

    def log_batch_verdicts(
        self,
        verdicts_by_check: dict[str, list[bool]],
        evaluated_layout_indices_by_check: dict[str, list[int]],
        required_checks: set[str] | None,
    ) -> None:
        """Log the check verdicts of the current batch of layouts, one layout at a time.

        A check is left off the layouts it did not run on, so a skipped one is not drawn as failed.

        Args:
            verdicts_by_check: Per check, its verdict for every layout of the batch.
            evaluated_layout_indices_by_check: Per check, which layouts of the batch it ran on.
            required_checks: Checks that gate acceptance; None means every check that ran gates it.
        """
        evaluated = {check: set(indices) for check, indices in evaluated_layout_indices_by_check.items()}
        for layout_index_within_batch, layout_index_across_batch in enumerate(self._layout_indices_across_batch):
            self.log_layout_verdicts(
                layout_index_across_batch,
                {
                    check: verdicts[layout_index_within_batch]
                    for check, verdicts in verdicts_by_check.items()
                    if layout_index_within_batch in evaluated[check]
                },
                required_checks,
            )

    def close(self) -> None:
        """Flush pending data and shut down the viewer window this run spawned."""
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
