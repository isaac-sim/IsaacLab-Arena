# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run cheap placement checks before expensive checks and collect candidate verdicts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_validation import PlacementValidationResults

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_validators import PlacementValidator
    from isaaclab_arena.relations.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class PlacementValidationPipeline:
    """Configured validators evaluated in cost order over solved candidate layouts."""

    def __init__(
        self,
        params: ObjectPlacerParams,
        validators: list[PlacementValidator],
        visualizer: PlacementRerunVisualizer | None = None,
    ):
        self.params = params
        self._validators = validators
        self._visualizer = visualizer

    def validate_candidates(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[PlacementValidationResults]:
        """Run every enabled validator over all candidates and collect per-candidate results.

        Each validator reports one verdict per candidate; the verdicts are transposed into one
        PlacementValidationResults per candidate, gated by the configured required_checks.

        Args:
            positions: Solved (x, y, z) per object, one dict per candidate.
            orientations: Absolute world Z-yaw per object, one dict per candidate (may be empty).
            bboxes: Per-object bboxes for each candidate's env, each (1, 3).
            collision_objects: Fixed background obstacles shared across candidates.
        """
        # required_checks=None means "every enabled check is required"; an empty set means no checks.
        required = self.params.required_checks
        num_candidates = len(positions)
        # Per check, which layouts of this batch (each refill) it actually ran on
        evaluated_layout_indices_by_check: dict[str, list[int]] = {}
        layout_pass_verdicts_by_check: dict[str, list[bool]] = {}

        if self._visualizer is not None:
            self._visualizer.start_new_batch(positions, orientations, bboxes)

        self._run_inexpensive_checks(
            positions,
            orientations,
            bboxes,
            collision_objects,
            layout_pass_verdicts_by_check,
            evaluated_layout_indices_by_check,
        )
        self._run_expensive_checks(
            positions,
            orientations,
            bboxes,
            collision_objects,
            required,
            layout_pass_verdicts_by_check,
            evaluated_layout_indices_by_check,
        )
        if self._visualizer is not None:
            self._visualizer.log_batch_verdicts(
                layout_pass_verdicts_by_check,
                evaluated_layout_indices_by_check,
                self.params.required_checks,
            )
        if layout_pass_verdicts_by_check:
            summary = ", ".join(
                f"{check}={sum(verdicts)}/{len(evaluated_layout_indices_by_check[check])}"
                for check, verdicts in layout_pass_verdicts_by_check.items()
            )
            print(f"[placement] Validated {num_candidates} candidate layout(s); passed per check: {summary}")
        return [
            PlacementValidationResults(
                validation_results={
                    check: verdicts[candidate_idx] for check, verdicts in layout_pass_verdicts_by_check.items()
                },
                required_checks=set(required) if required is not None else None,
            )
            for candidate_idx in range(len(positions))
        ]

    def _run_inexpensive_checks(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        evaluated_layout_indices_by_check: dict[str, list[int]],
    ) -> None:
        """Run every inexpensive validator on all candidates, recording verdicts and evaluated layouts."""
        num_candidates = len(positions)
        for validator in self._validators:
            if not validator.run_after_inexpensive_checks:
                layout_pass_verdicts_by_check[validator.check] = validator.validate_batch(
                    positions, orientations, bboxes, collision_objects
                )
                evaluated_layout_indices_by_check[validator.check] = list(range(num_candidates))

    def _run_expensive_checks(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
        required: set[str] | None,
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        evaluated_layout_indices_by_check: dict[str, list[int]],
    ) -> None:
        """Run each expensive validator only on candidates that passed the required inexpensive checks."""
        num_candidates = len(positions)
        for validator in self._validators:
            if validator.run_after_inexpensive_checks:
                passed_layout_indices = [
                    i
                    for i in range(num_candidates)
                    if self._passes_required_checks(layout_pass_verdicts_by_check, required, i)
                ]
                if self._visualizer is not None:
                    self._visualizer.set_active_layouts(passed_layout_indices)
                # only passed layouts are validated
                verdicts_over_passed_layout = validator.validate_batch(
                    [positions[i] for i in passed_layout_indices],
                    [orientations[i] for i in passed_layout_indices],
                    [bboxes[i] for i in passed_layout_indices],
                    collision_objects,
                )
                verdicts = [False] * num_candidates
                for layout_index_within_batch, verdict in zip(passed_layout_indices, verdicts_over_passed_layout):
                    verdicts[layout_index_within_batch] = verdict
                layout_pass_verdicts_by_check[validator.check] = verdicts
                evaluated_layout_indices_by_check[validator.check] = passed_layout_indices

    @staticmethod
    def _passes_required_checks(
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        required_checks: set[str] | None,
        candidate_idx: int,
    ) -> bool:
        """Whether a candidate passes every required check computed so far.

        required_checks=None means every computed check is required; an explicit set gates only its members.
        """
        for check, verdicts in layout_pass_verdicts_by_check.items():
            is_required = required_checks is None or check in required_checks
            if is_required and not verdicts[candidate_idx]:
                return False
        return True
