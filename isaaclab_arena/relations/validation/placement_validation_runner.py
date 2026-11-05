# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run cheap placement checks before expensive checks and collect candidate verdicts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.relations.optimization.placement_candidate_batch import PlacementCandidateBatch
from isaaclab_arena.relations.placement_diagnostics import (
    BatchValidationDiagnostics,
    RejectionLog,
    ValidationCheckDiagnostics,
)
from isaaclab_arena.relations.validation.types import PlacementValidationResults

if TYPE_CHECKING:
    from isaaclab_arena.relations.geometry.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.validation.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator


class _BatchCheckResults:
    """Verdicts, evaluated layouts, and rejections of every check run on one batch of candidate layouts."""

    def __init__(self, required_checks: set[str] | None) -> None:
        """Start with no checks run.

        Args:
            required_checks: Checks a candidate must pass to be valid. None requires every check; an empty set
                requires none.
        """
        self._required_checks = required_checks
        self._verdicts_by_check: dict[str, list[bool]] = {}
        self._evaluated_layout_indices_by_check: dict[str, list[int]] = {}
        self._rejections_by_check: dict[str, RejectionLog] = {}

    def add(
        self, check: str, verdicts: list[bool], evaluated_layout_indices: list[int], rejections: RejectionLog
    ) -> None:
        """Add one check's results.

        Args:
            check: Name of the check.
            verdicts: Verdict per candidate in batch order. Candidates the check skipped are marked failed.
            evaluated_layout_indices: Batch positions of the candidates the check ran on.
            rejections: Causes of the check's rejections.
        """
        assert check not in self._verdicts_by_check, f"Check '{check}' already has results for this batch."
        self._verdicts_by_check[check] = verdicts
        self._evaluated_layout_indices_by_check[check] = evaluated_layout_indices
        self._rejections_by_check[check] = rejections

    def is_required(self, check: str) -> bool:
        """Whether a candidate must pass check to be valid."""
        return self._required_checks is None or check in self._required_checks

    def passes_required_checks(self, candidate_idx: int) -> bool:
        """Whether the candidate at batch position candidate_idx passes every required check added so far."""
        return all(
            verdicts[candidate_idx] for check, verdicts in self._verdicts_by_check.items() if self.is_required(check)
        )

    def get_validation_results(self, candidate_idx: int) -> PlacementValidationResults:
        """Return the verdict of every check for the candidate at batch position candidate_idx."""
        return PlacementValidationResults(
            validation_results={check: verdicts[candidate_idx] for check, verdicts in self._verdicts_by_check.items()},
            required_checks=set(self._required_checks) if self._required_checks is not None else None,
        )

    def get_verdicts_by_check(self) -> dict[str, list[bool]]:
        """Return each check's verdict per candidate, in the order the checks ran."""
        return dict(self._verdicts_by_check)

    def get_evaluated_layout_indices_by_check(self) -> dict[str, list[int]]:
        """Return the batch positions each check ran on, in the order the checks ran."""
        return dict(self._evaluated_layout_indices_by_check)

    def get_check_diagnostics(self) -> dict[str, ValidationCheckDiagnostics]:
        """Return each check's evaluated, passed, and rejection counts, in the order the checks ran."""
        check_diagnostics = {}
        for check, verdicts in self._verdicts_by_check.items():
            # A check marks the candidates it skipped as failed, so its passes all come from candidates it ran on.
            check_diagnostics[check] = ValidationCheckDiagnostics(
                num_evaluated=len(self._evaluated_layout_indices_by_check[check]),
                num_passed=sum(verdicts),
                required=self.is_required(check),
                num_rejected_by_cause=self._rejections_by_check[check].num_rejected_by_cause,
            )
        return check_diagnostics


class PlacementValidationRunner:
    """Run configured placement checks and collect each candidate's results.

    Geometry checks run first. Validators marked run_after_inexpensive_checks, such as IK,
    run only on candidates that pass the required earlier checks. Neither pass steps physics.
    """

    def __init__(
        self,
        params: ObjectPlacerParams,
        validators: list[PrePhysicsPlacementValidator],
        visualizer: PlacementRerunVisualizer | None = None,
    ):
        self.params = params
        self._validators = validators
        self._visualizer = visualizer

    def validate_candidates(
        self,
        batch: PlacementCandidateBatch,
        collision_objects: list[CollisionObject],
    ) -> BatchValidationDiagnostics:
        """Update each candidate's validation in place using the configured required-check policy.

        Returns:
            The batch's validation diagnostics.
        """
        num_candidates = len(batch)

        if self._visualizer is not None:
            self._visualizer.start_new_batch(
                [candidate.positions for candidate in batch.candidates],
                [candidate.orientations for candidate in batch.candidates],
                [candidate.bboxes for candidate in batch.candidates],
            )

        results = _BatchCheckResults(self.params.required_checks)
        self._run_inexpensive_checks(batch, collision_objects, results)
        self._run_expensive_checks(batch, collision_objects, results)
        if self._visualizer is not None:
            self._visualizer.log_batch_verdicts(
                results.get_verdicts_by_check(),
                results.get_evaluated_layout_indices_by_check(),
                self.params.required_checks,
            )
        for candidate_idx, candidate in enumerate(batch.candidates):
            candidate.validation = results.get_validation_results(candidate_idx)
        batch_diagnostics = BatchValidationDiagnostics(
            num_candidates=num_candidates,
            num_valid=sum(
                candidate.validation.do_all_required_validation_checks_pass() for candidate in batch.candidates
            ),
            checks=results.get_check_diagnostics(),
        )
        if batch_diagnostics.checks:
            print(
                f"[placement] Validated {num_candidates} candidate layout(s); passed per check:"
                f" {batch_diagnostics.format_passed_per_check()}"
            )
        return batch_diagnostics

    def _run_inexpensive_checks(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], results: _BatchCheckResults
    ) -> None:
        """Run every inexpensive validator on all candidates and add its results to results."""
        for validator in self._validators:
            if not validator.run_after_inexpensive_checks:
                rejections = RejectionLog(validator.check, self.params.verbose)
                verdicts = validator.validate_batch(batch, collision_objects, rejections)
                results.add(validator.check, verdicts, list(range(len(batch))), rejections)

    def _run_expensive_checks(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject], results: _BatchCheckResults
    ) -> None:
        """Run each expensive validator on the candidates that pass every required check in results, and add its
        results to results."""
        num_candidates = len(batch)
        for validator in self._validators:
            if validator.run_after_inexpensive_checks:
                passed_layout_indices = [i for i in range(num_candidates) if results.passes_required_checks(i)]
                if self._visualizer is not None:
                    self._visualizer.set_active_layouts(passed_layout_indices)
                rejections = RejectionLog(validator.check, self.params.verbose)
                # only passed layouts are validated
                verdicts_over_passed_layout = validator.validate_batch(
                    batch.select(passed_layout_indices), collision_objects, rejections
                )
                verdicts = [False] * num_candidates
                for layout_index_within_batch, verdict in zip(
                    passed_layout_indices, verdicts_over_passed_layout, strict=True
                ):
                    verdicts[layout_index_within_batch] = verdict
                results.add(validator.check, verdicts, passed_layout_indices, rejections)
