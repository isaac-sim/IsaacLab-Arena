# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run cheap placement checks before expensive checks and collect candidate verdicts."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _CheckRun:
    """What one check did on one batch of candidate layouts."""

    verdicts: list[bool]
    """Verdict per candidate in batch order. Candidates the check skipped are marked failed."""

    evaluated_layout_indices: list[int]
    """Batch positions of the candidates the check ran on."""

    rejections: RejectionLog
    """Causes of the check's rejections."""


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
        # required_checks=None means "every enabled check is required"; an empty set means no checks are required.
        required = self.params.required_checks
        num_candidates = len(batch)

        if self._visualizer is not None:
            self._visualizer.start_new_batch(
                [candidate.positions for candidate in batch.candidates],
                [candidate.orientations for candidate in batch.candidates],
                [candidate.bboxes for candidate in batch.candidates],
            )

        runs_by_check = self._run_inexpensive_checks(batch, collision_objects)
        runs_by_check = self._run_expensive_checks(batch, collision_objects, required, runs_by_check)
        if self._visualizer is not None:
            self._visualizer.log_batch_verdicts(
                {check: run.verdicts for check, run in runs_by_check.items()},
                {check: run.evaluated_layout_indices for check, run in runs_by_check.items()},
                self.params.required_checks,
            )
        for candidate_idx, candidate in enumerate(batch.candidates):
            candidate.validation = PlacementValidationResults(
                validation_results={check: run.verdicts[candidate_idx] for check, run in runs_by_check.items()},
                required_checks=set(required) if required is not None else None,
            )
        batch_diagnostics = self._build_batch_validation_diagnostics(batch, required, runs_by_check)
        if batch_diagnostics.checks:
            print(
                f"[placement] Validated {num_candidates} candidate layout(s); passed per check:"
                f" {batch_diagnostics.format_passed_per_check()}"
            )
        return batch_diagnostics

    def _run_inexpensive_checks(
        self, batch: PlacementCandidateBatch, collision_objects: list[CollisionObject]
    ) -> dict[str, _CheckRun]:
        """Return each inexpensive validator's run on all candidates, keyed by check."""
        runs_by_check = {}
        for validator in self._validators:
            if not validator.run_after_inexpensive_checks:
                rejections = RejectionLog(validator.check, self.params.verbose)
                verdicts = validator.validate_batch(batch, collision_objects, rejections)
                runs_by_check[validator.check] = _CheckRun(verdicts, list(range(len(batch))), rejections)
        return runs_by_check

    def _run_expensive_checks(
        self,
        batch: PlacementCandidateBatch,
        collision_objects: list[CollisionObject],
        required: set[str] | None,
        runs_by_check: dict[str, _CheckRun],
    ) -> dict[str, _CheckRun]:
        """Return runs_by_check with each expensive validator's run added, keyed by check.

        Each expensive validator runs only on candidates that passed every required check run before it.
        """
        runs_by_check = dict(runs_by_check)
        num_candidates = len(batch)
        for validator in self._validators:
            if validator.run_after_inexpensive_checks:
                passed_layout_indices = [
                    i for i in range(num_candidates) if self._passes_required_checks(runs_by_check, required, i)
                ]
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
                runs_by_check[validator.check] = _CheckRun(verdicts, passed_layout_indices, rejections)
        return runs_by_check

    @staticmethod
    def _build_batch_validation_diagnostics(
        batch: PlacementCandidateBatch,
        required: set[str] | None,
        runs_by_check: dict[str, _CheckRun],
    ) -> BatchValidationDiagnostics:
        """Return the batch's validation diagnostics from each check's run.

        Call it after every candidate's validation results are set, since the valid count reads them.
        """
        checks = {}
        for check, run in runs_by_check.items():
            # A check marks the candidates it skipped as failed, so its passes all come from candidates it ran on.
            checks[check] = ValidationCheckDiagnostics(
                num_evaluated=len(run.evaluated_layout_indices),
                num_passed=sum(run.verdicts),
                required=required is None or check in required,
                num_rejected_by_cause=run.rejections.num_rejected_by_cause,
            )
        return BatchValidationDiagnostics(
            num_candidates=len(batch),
            num_valid=sum(
                candidate.validation.do_all_required_validation_checks_pass() for candidate in batch.candidates
            ),
            checks=checks,
        )

    @staticmethod
    def _passes_required_checks(
        runs_by_check: dict[str, _CheckRun],
        required_checks: set[str] | None,
        candidate_idx: int,
    ) -> bool:
        """Whether a candidate passes every required check in runs_by_check.

        required_checks=None means every check run is required; an explicit set gates only its members.
        """
        for check, run in runs_by_check.items():
            is_required = required_checks is None or check in required_checks
            if is_required and not run.verdicts[candidate_idx]:
                return False
        return True
