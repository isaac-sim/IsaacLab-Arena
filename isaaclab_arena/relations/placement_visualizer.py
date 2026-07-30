# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rerun debug view of build-time placement validation, sim-free (no SimApp).

Rerun's viewer is a separate process fed by the logging SDK, so nothing here touches Isaac Sim -- the
window comes up while layouts are being solved, before any simulation exists. Enabled through
``ObjectPlacerParams.debug_visualize`` / ``debug_visualize_rrd_path``.

Every candidate layout is one frame of the ``candidate`` timeline, so scrubbing it shows what was
solved and which checks rejected it. Checks that know more about a candidate than its boxes add their
own layer under ``world/robot`` (see the cuRobo reachability check).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

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

ACCEPTED_COLOR = (40, 200, 80)
"""Color marking a candidate every required check accepted."""

REJECTED_COLOR = (220, 50, 50)
"""Color marking a candidate at least one check rejected."""

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


class PlacementRerunVisualizer:
    """Streams every validated candidate layout to Rerun, one frame per candidate.

    Holds only plain attributes (the recording itself is Rerun's process-global stream) so the
    placement event config can deep-copy the pool that owns it.
    """

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
        if spawn:
            # connect=False: the sink is set below, so spawn only has to bring the viewer process up.
            rr.spawn(connect=False, executable_path=find_rerun_viewer_executable())
            sinks.append(rr.GrpcSink())
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

    def log_verdicts(self, candidate_index: int, verdicts_by_check: dict[str, bool]) -> None:
        """Log which checks accepted one candidate, as a text line and an accepted/rejected marker.

        Args:
            candidate_index: Timeline index to log against.
            verdicts_by_check: Per-check verdict for this candidate.
        """
        import rerun as rr

        self.set_time(candidate_index)
        failed = [check for check, passed in verdicts_by_check.items() if not passed]
        accepted = not failed
        rr.log(
            f"{LAYOUT_ENTITY}/verdict",
            rr.TextLog(
                f"candidate {candidate_index}: {'accepted' if accepted else 'rejected'}"
                + (f" (failed: {', '.join(failed)})" if failed else ""),
                level=rr.TextLogLevel.INFO if accepted else rr.TextLogLevel.WARN,
            ),
        )
        for check, passed in verdicts_by_check.items():
            rr.log(f"checks/{check}", rr.Scalars(float(passed)))

    def close(self) -> None:
        """Flush pending data so a recorded ``.rrd`` is readable without waiting for interpreter exit.

        A spawned viewer keeps running afterwards, so the layouts stay inspectable.
        """
        import rerun as rr

        rr.get_global_data_recording().flush()
