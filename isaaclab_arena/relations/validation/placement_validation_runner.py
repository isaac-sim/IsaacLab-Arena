# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run cheap placement checks before expensive checks and collect candidate verdicts."""

from __future__ import annotations

from collections.abc import Iterator
from math import inf
from typing import TYPE_CHECKING

from isaaclab_arena.relations.optimization.placement_candidate_batch import PlacementCandidateBatch
from isaaclab_arena.relations.placement_diagnostics import (
    BatchValidationDiagnostics,
    Rejection,
    ValidationCheckDiagnostics,
)
from isaaclab_arena.relations.validation.types import PlacementValidationResults

if TYPE_CHECKING:
    from isaaclab_arena.relations.geometry.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.validation.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator


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
        required_valid_per_env: int | None = None,
    ) -> BatchValidationDiagnostics:
        """Update each candidate's validation in place using the configured required-check policy.

        Args:
            batch: Candidate layouts to check; every candidate carries its env and its solver loss.
            collision_objects: Fixed background obstacles in the scene.
            required_valid_per_env: When set, the expensive checks run in loss order within each env and stop
                once that many layouts of the env passed every required check; the layouts they never reached
                are recorded as failing those checks, which keeps them out of the valid set. None checks every
                candidate that passed the inexpensive checks.

        Returns:
            The batch's validation diagnostics.
        """
        # required_checks=None means "every enabled check is required"; an empty set means no checks are required.
        required = self.params.required_checks
        num_candidates = len(batch)
        # Per check, which layouts of this batch (each refill) it actually ran on
        evaluated_layout_indices_by_check: dict[str, list[int]] = {}
        layout_pass_verdicts_by_check: dict[str, list[bool]] = {}
        rejections_by_check: dict[str, list[Rejection]] = {validator.check: [] for validator in self._validators}

        if self._visualizer is not None:
            self._visualizer.start_new_batch(
                [candidate.positions for candidate in batch.candidates],
                [candidate.orientations for candidate in batch.candidates],
                [candidate.bboxes for candidate in batch.candidates],
            )

        self._run_inexpensive_checks(
            batch,
            collision_objects,
            layout_pass_verdicts_by_check,
            evaluated_layout_indices_by_check,
            rejections_by_check,
        )
        self._run_expensive_checks(
            batch,
            collision_objects,
            required,
            layout_pass_verdicts_by_check,
            evaluated_layout_indices_by_check,
            rejections_by_check,
            required_valid_per_env,
        )
        if self._visualizer is not None:
            self._visualizer.log_batch_verdicts(
                layout_pass_verdicts_by_check,
                evaluated_layout_indices_by_check,
                self.params.required_checks,
            )
        for candidate_idx, candidate in enumerate(batch.candidates):
            candidate.validation = PlacementValidationResults(
                validation_results={
                    check: verdicts[candidate_idx] for check, verdicts in layout_pass_verdicts_by_check.items()
                },
                required_checks=set(required) if required is not None else None,
            )
        batch_diagnostics = self._build_batch_validation_diagnostics(
            batch, required, layout_pass_verdicts_by_check, evaluated_layout_indices_by_check, rejections_by_check
        )
        if batch_diagnostics.checks:
            print(
                f"[placement] Validated {num_candidates} candidate layout(s); passed per check:"
                f" {batch_diagnostics.format_passed_per_check()}"
            )
            self._print_rejection_causes(batch_diagnostics)
        return batch_diagnostics

    def _run_inexpensive_checks(
        self,
        batch: PlacementCandidateBatch,
        collision_objects: list[CollisionObject],
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        evaluated_layout_indices_by_check: dict[str, list[int]],
        rejections_by_check: dict[str, list[Rejection]],
    ) -> None:
        """Run every inexpensive validator on all candidates, recording verdicts, evaluated layouts, and rejections."""
        num_candidates = len(batch)
        for validator in self._validators:
            if not validator.run_after_inexpensive_checks:
                rejections = rejections_by_check[validator.check]
                verdicts = validator.validate_batch(batch, collision_objects, rejections)
                layout_pass_verdicts_by_check[validator.check] = verdicts
                evaluated_layout_indices_by_check[validator.check] = list(range(num_candidates))
                self._print_rejections(validator.check, rejections)

    def _run_expensive_checks(
        self,
        batch: PlacementCandidateBatch,
        collision_objects: list[CollisionObject],
        required: set[str] | None,
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        evaluated_layout_indices_by_check: dict[str, list[int]],
        rejections_by_check: dict[str, list[Rejection]],
        required_valid_per_env: int | None = None,
    ) -> None:
        """Run each expensive validator only on candidates that passed the required checks so far.

        With ``required_valid_per_env`` the candidates of each env are checked in loss order, in rounds that
        request for every env only the layouts it still lacks, until every env has that many passing layouts or
        runs out of candidates; unchecked candidates fail the check. Without it every passing candidate is checked.
        """
        num_candidates = len(batch)
        for validator in self._validators:
            if not validator.run_after_inexpensive_checks:
                continue
            passed_layout_indices = [
                i
                for i in range(num_candidates)
                if self._passes_required_checks(layout_pass_verdicts_by_check, required, i)
            ]
            if self._visualizer is not None:
                self._visualizer.set_active_layouts(passed_layout_indices)
            rejections = rejections_by_check[validator.check]
            verdicts = [False] * num_candidates
            if required_valid_per_env is None:
                rounds = [passed_layout_indices]
            else:
                rounds = self._layout_rounds_until_enough_valid(
                    batch, passed_layout_indices, required_valid_per_env, verdicts
                )
            evaluated_layout_indices: list[int] = []
            for round_layout_indices in rounds:
                round_verdicts = validator.validate_batch(
                    batch.select(round_layout_indices), collision_objects, rejections
                )
                for layout_index, verdict in zip(round_layout_indices, round_verdicts, strict=True):
                    verdicts[layout_index] = verdict
                evaluated_layout_indices.extend(round_layout_indices)
            layout_pass_verdicts_by_check[validator.check] = verdicts
            evaluated_layout_indices_by_check[validator.check] = sorted(evaluated_layout_indices)
            self._print_rejections(validator.check, rejections)

    @staticmethod
    def _layout_rounds_until_enough_valid(
        batch: PlacementCandidateBatch,
        passed_layout_indices: list[int],
        required_valid_per_env: int,
        verdicts: list[bool],
    ) -> Iterator[list[int]]:
        """Yield rounds of layout indices to check, in loss order per env, until every env has enough valid ones.

        Reads ``verdicts`` between rounds, so the caller must write each round's verdicts into it before
        asking for the next round.
        """
        ordered_per_env: dict[int, list[int]] = {}
        for layout_index in passed_layout_indices:
            ordered_per_env.setdefault(batch.candidates[layout_index].env_id, []).append(layout_index)
        for layout_indices in ordered_per_env.values():
            layout_indices.sort(key=lambda i: batch.candidates[i].loss if batch.candidates[i].loss is not None else inf)
        next_position = dict.fromkeys(ordered_per_env, 0)
        while True:
            round_layout_indices: list[int] = []
            for env_id, layout_indices in ordered_per_env.items():
                checked = layout_indices[: next_position[env_id]]
                missing = required_valid_per_env - sum(verdicts[i] for i in checked)
                if missing <= 0:
                    continue
                take = layout_indices[next_position[env_id] : next_position[env_id] + missing]
                next_position[env_id] += len(take)
                round_layout_indices.extend(take)
            if not round_layout_indices:
                return
            yield round_layout_indices

    @staticmethod
    def _print_rejection_causes(batch_diagnostics: BatchValidationDiagnostics) -> None:
        """Print one line per rejection cause of each check, with its count and the first rejection's measured
        violation, such as "  next_to: 'bowl' next_to(plate, positive_x): on the wrong side x12 (e.g. by 0.150 m)"."""
        for check, check_diagnostics in batch_diagnostics.checks.items():
            first_rejection_by_cause: dict[str, Rejection] = {}
            for rejection in check_diagnostics.rejections:
                first_rejection_by_cause.setdefault(rejection.cause, rejection)
            for cause, count in check_diagnostics.get_num_rejected_by_cause().items():
                detail = first_rejection_by_cause[cause].detail
                example = f" (e.g. {detail})" if detail else ""
                print(f"  {check}: {cause} x{count}{example}")

    def _print_rejections(self, check: str, rejections: list[Rejection]) -> None:
        """Print a check's rejections, grouped by cause, when verbose.

        Each printed rejection has its own line, such as
        "  [next_to] 'bowl' next_to(plate, positive_x): on the wrong side (by 0.150 m)". At most
        max_printed_rejections_per_cause are printed for each cause, and one more line counts the rest.
        """
        if not self.params.verbose:
            return
        rejections_by_cause: dict[str, list[Rejection]] = {}
        for rejection in rejections:
            rejections_by_cause.setdefault(rejection.cause, []).append(rejection)
        max_printed = self.params.max_printed_rejections_per_cause
        for cause, cause_rejections in rejections_by_cause.items():
            printed_rejections = cause_rejections if max_printed is None else cause_rejections[:max_printed]
            for rejection in printed_rejections:
                print(f"  [{check}] {rejection.format()}")
            num_unprinted = len(cause_rejections) - len(printed_rejections)
            if num_unprinted > 0:
                print(f"  [{check}] {cause}: {num_unprinted} more")

    @staticmethod
    def _build_batch_validation_diagnostics(
        batch: PlacementCandidateBatch,
        required: set[str] | None,
        layout_pass_verdicts_by_check: dict[str, list[bool]],
        evaluated_layout_indices_by_check: dict[str, list[int]],
        rejections_by_check: dict[str, list[Rejection]],
    ) -> BatchValidationDiagnostics:
        """Return the batch's validation diagnostics from each check's verdicts, evaluated layouts, and rejections.

        Call it after every candidate's validation results are set, since the valid count reads them.
        """
        checks = {}
        for check, verdicts in layout_pass_verdicts_by_check.items():
            # A check marks the candidates it skipped as failed, so its passes all come from candidates it ran on.
            checks[check] = ValidationCheckDiagnostics(
                num_evaluated=len(evaluated_layout_indices_by_check[check]),
                num_passed=sum(verdicts),
                required=required is None or check in required,
                rejections=rejections_by_check[check],
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
