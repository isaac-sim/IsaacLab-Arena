# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rerun debug view of build-time placement validation.

Placement is sim-free, so these run a real solve and a real recording headless: the ``.rrd`` sink
stands in for the viewer window, and no Isaac Sim or GPU is involved.
"""

from __future__ import annotations

import subprocess

import pytest

from isaaclab_arena.relations import placement_visualizer
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_visualizer import (
    PlacementRerunVisualizer,
    get_active_placement_visualizer,
    summarize_candidate_verdict,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

MAX_PLACEMENT_ATTEMPTS = 3
"""Candidate layouts solved per placement, i.e. the number of frames one place() should draw."""


@pytest.fixture(autouse=True)
def _fresh_process_visualizer(monkeypatch):
    """Drop the process-wide view between tests so each one gets its own recording and ``.rrd``."""
    monkeypatch.setattr(placement_visualizer, "_ACTIVE_VISUALIZER", None)


def _desk_and_box() -> list[DummyObject]:
    """A desk anchor with a box placed on it -- the smallest layout with something to solve."""
    desk = DummyObject(
        name="desk",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    box = DummyObject(
        name="box",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.2)),
    )
    box.add_relation(On(desk, clearance_m=0.01))
    return [desk, box]


def _placer_params(**overrides) -> ObjectPlacerParams:
    """Placement params for a small, deterministic solve."""
    return ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=200, convergence_threshold=1e-3),
        apply_positions_to_objects=False,
        max_placement_attempts=MAX_PLACEMENT_ATTEMPTS,
        placement_seed=5,
        **overrides,
    )


def test_placement_has_no_debug_view_by_default():
    """The debug view is opt-in, so a default placement never touches Rerun."""
    ObjectPlacer(_placer_params())

    assert get_active_placement_visualizer() is None


def test_placement_records_every_candidate_layout(tmp_path):
    """Recording to an .rrd draws every candidate the solve produced, one frame each, without a viewer."""
    output_path = tmp_path / "placement.rrd"
    placer = ObjectPlacer(_placer_params(debug_visualize_output_path=str(output_path)))

    placer.place(_desk_and_box(), num_envs=1)
    visualizer = get_active_placement_visualizer()
    visualizer.close()

    assert visualizer.num_logged_candidates == MAX_PLACEMENT_ATTEMPTS
    assert output_path.is_file() and output_path.stat().st_size > 0


def test_placement_shares_one_debug_view_across_placers(tmp_path):
    """Every placer in the process draws into the same view, keeping one viewer and one timeline."""
    ObjectPlacer(_placer_params(debug_visualize_output_path=str(tmp_path / "placement.rrd")))
    view_of_first_placer = get_active_placement_visualizer()

    ObjectPlacer(_placer_params(debug_visualize_output_path=str(tmp_path / "ignored.rrd")))

    assert get_active_placement_visualizer() is view_of_first_placer


class _FakeViewerProcess:
    """Stands in for the spawned viewer window, so the test needs no display.

    Args:
        wedged: Whether the window ignores SIGTERM, i.e. whether the first wait() times out.
    """

    def __init__(self, wedged: bool = False) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self._wedged = wedged

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        if self._wedged and self.kill_calls == 0:
            raise subprocess.TimeoutExpired(cmd="rerun", timeout=timeout)
        return 0


def test_closing_the_view_shuts_down_the_viewer_it_spawned(tmp_path, monkeypatch):
    """The window belongs to the run, so closing the view takes it down instead of leaving it on the port."""
    import rerun as rr

    viewer = _FakeViewerProcess()
    # Record where the viewer's live stream would go, so the test needs neither a display nor a port.
    monkeypatch.setattr(
        placement_visualizer,
        "spawn_viewer_process",
        lambda: (viewer, rr.FileSink(str(tmp_path / "viewer_stand_in.rrd"))),
    )
    visualizer = PlacementRerunVisualizer(app_id="arena_test", spawn=True, output_path=str(tmp_path / "p.rrd"))

    visualizer.close()
    visualizer.close()

    assert viewer.terminate_calls == 1


def test_closing_a_wedged_viewer_falls_back_to_killing_it(tmp_path, monkeypatch):
    """A window that ignores SIGTERM still has to go, or it outlives the run holding the viewer port."""
    import rerun as rr

    viewer = _FakeViewerProcess(wedged=True)
    monkeypatch.setattr(
        placement_visualizer,
        "spawn_viewer_process",
        lambda: (viewer, rr.FileSink(str(tmp_path / "viewer_stand_in.rrd"))),
    )
    visualizer = PlacementRerunVisualizer(app_id="arena_test", spawn=True, output_path=str(tmp_path / "p.rrd"))

    visualizer.close()

    assert viewer.kill_calls == 1


def test_only_required_checks_reject_a_candidate():
    """The view has to agree with the placer, which gates layouts on required_checks alone."""
    verdicts = {"no_overlap": True, "ik_reachable": False}

    message, accepted = summarize_candidate_verdict(3, verdicts, required_checks={"no_overlap"})

    assert accepted
    assert message == "candidate 3: accepted (failed but not required: ik_reachable)"


def test_a_failed_required_check_rejects_a_candidate():
    """A failure the placer does gate on reads as a rejection, naming what blocked it."""
    verdicts = {"no_overlap": False, "ik_reachable": False}

    message, accepted = summarize_candidate_verdict(3, verdicts, required_checks={"no_overlap"})

    assert not accepted
    assert message == "candidate 3: rejected (failed: no_overlap)"


def test_every_check_gates_a_candidate_when_none_are_named_required():
    """required_checks=None means every check that ran is required, matching ObjectPlacerParams."""
    verdicts = {"no_overlap": True, "ik_reachable": False}

    message, accepted = summarize_candidate_verdict(3, verdicts, required_checks=None)

    assert not accepted
    assert message == "candidate 3: rejected (failed: ik_reachable)"


def test_a_check_that_skipped_a_candidate_is_not_drawn_as_rejecting_it(tmp_path, monkeypatch):
    """Expensive checks only see candidates the cheap ones passed; the rest are unevaluated, not failed."""
    visualizer = PlacementRerunVisualizer(app_id="arena_test", spawn=False, output_path=str(tmp_path / "p.rrd"))
    drawn_verdicts: dict[int, dict[str, bool]] = {}
    monkeypatch.setattr(
        visualizer,
        "log_verdicts",
        lambda candidate_index, verdicts_by_check, required_checks: drawn_verdicts.update(
            {candidate_index: verdicts_by_check}
        ),
    )

    visualizer.log_verdict_batch(
        candidate_indices=[0, 1],
        verdicts_by_check={"no_overlap": [True, False], "ik_reachable": [True, False]},
        evaluated_batch_indices_by_check={"no_overlap": [0, 1], "ik_reachable": [0]},
        required_checks=None,
    )

    assert drawn_verdicts == {
        0: {"no_overlap": True, "ik_reachable": True},
        1: {"no_overlap": False},
    }


def test_candidate_frames_keep_counting_across_batches(tmp_path):
    """A pool that refills gets fresh frames, so a later batch does not overwrite an earlier one."""
    visualizer = PlacementRerunVisualizer(app_id="arena_test", spawn=False, output_path=str(tmp_path / "p.rrd"))

    assert visualizer.next_batch_indices(3) == [0, 1, 2]
    assert visualizer.next_batch_indices(2) == [3, 4]


def test_active_candidates_map_batch_position_to_frame(tmp_path):
    """A check that only ran on some candidates still resolves each one's own frame."""
    visualizer = PlacementRerunVisualizer(app_id="arena_test", spawn=False, output_path=str(tmp_path / "p.rrd"))

    visualizer.set_active_candidates([1, 4])

    assert visualizer.candidate_index_for_batch_index(0) == 1
    assert visualizer.candidate_index_for_batch_index(1) == 4
