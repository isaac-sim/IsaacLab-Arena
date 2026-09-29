# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configurable checks of measured poses after physics."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.validation.base import PlacementValidator
from isaaclab_arena.relations.validation.types import PlacementValidatorReport

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.offline_placement.settled_batch import SettledBatch
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass
class PostPhysicsPlacementValidator(PlacementValidator):
    """A configured check returning one report per candidate environment."""

    stage: ClassVar[str] = "post_physics"
    enabled: bool = True
    """Whether this check must pass for applicable candidates."""

    @abstractmethod
    def validate(self, data: SettledBatch) -> list[PlacementValidatorReport]:
        """Return one report per candidate environment, in env_ids order."""

    def get_geometry_keys(self, assets: Sequence[PlaceableAsset]) -> set[str]:
        """Return scene assets whose bounds and poses this check needs captured."""
        return set()

    def configuration(self) -> dict:
        """Return the implementation path and effective settings."""
        return {"_target_": f"{type(self).__module__}.{type(self).__qualname__}", **asdict(self)}

    def skip_reason(self, articulation_keys: list[str]) -> str | None:
        """Return why the check is disabled or inapplicable, otherwise None."""
        return None if self.enabled else "disabled by configuration"

    def report(self, passed: bool | None, reason: str = "") -> PlacementValidatorReport:
        """Describe the check's configuration and outcome."""
        return PlacementValidatorReport(
            check=self.check,
            stage=self.stage,
            configuration=self.configuration(),
            passed=passed,
            reason=reason,
        )


@dataclass
class VelocityValidator(PostPhysicsPlacementValidator):
    """Require final linear and angular root speeds below the configured limits."""

    check: ClassVar[str] = "physics_settled"
    lin_vel_thresh: float = PhysicsSettleParams.lin_vel_thresh
    """Maximum final root linear speed, in m/s."""
    ang_vel_thresh: float = PhysicsSettleParams.ang_vel_thresh
    """Maximum final root angular speed, in rad/s."""

    def __post_init__(self) -> None:
        assert (
            math.isfinite(self.lin_vel_thresh) and self.lin_vel_thresh >= 0
        ), "lin_vel_thresh must be finite and non-negative"
        assert (
            math.isfinite(self.ang_vel_thresh) and self.ang_vel_thresh >= 0
        ), "ang_vel_thresh must be finite and non-negative"

    def validate(self, data: SettledBatch) -> list[PlacementValidatorReport]:
        import torch

        settled_per_root = []
        for velocity in data.final_root_velocities.values():
            linear_speed = torch.linalg.vector_norm(velocity[:, :3], dim=-1)
            angular_speed = torch.linalg.vector_norm(velocity[:, 3:], dim=-1)
            settled_per_root.append((linear_speed < self.lin_vel_thresh) & (angular_speed < self.ang_vel_thresh))
        assert settled_per_root, "Velocity validation requires captured root velocities"
        settled = torch.stack(settled_per_root).all(dim=0)[data.env_ids].tolist()
        return [self.report(passed, "" if passed else "objects exceed final velocity limits") for passed in settled]


@dataclass
class PoseShiftValidator(PostPhysicsPlacementValidator):
    """Limit initial-to-final root displacement and rotation."""

    check: ClassVar[str] = "pose_shift"
    max_translation_m: float = 0.002
    """Maximum displacement, in metres."""
    max_rotation_deg: float = 2.0
    """Maximum orientation change, in degrees."""

    def __post_init__(self) -> None:
        assert (
            math.isfinite(self.max_translation_m) and self.max_translation_m >= 0
        ), "max_translation_m must be finite and non-negative"
        assert (
            math.isfinite(self.max_rotation_deg) and self.max_rotation_deg >= 0
        ), "max_rotation_deg must be finite and non-negative"

    def validate(self, data: SettledBatch) -> list[PlacementValidatorReport]:
        return self._validate_poses(data.env_ids, data.initial_root_poses, data.final_root_poses)

    def _validate_poses(
        self, env_ids: list[int], initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor]
    ) -> list[PlacementValidatorReport]:
        from isaaclab_arena.utils.physics_settle import get_pose_drift

        reports = []
        for env_id in env_ids:
            reason = ""
            for key, poses in final.items():
                drift = get_pose_drift(initial[key][env_id], poses[env_id])
                if drift is None:
                    reason = f"{key}: non-finite pose"
                    break
                distance, angle = drift
                if distance > self.max_translation_m or angle > self.max_rotation_deg:
                    reason = (
                        f"{key}: moved {distance:.6f} m and rotated {angle:.3f} deg; "
                        f"limits {self.max_translation_m:g} m, {self.max_rotation_deg:g} deg"
                    )
                    break
            reports.append(self.report(not bool(reason), reason))
        return reports


@dataclass
class ArticulationLinkShiftValidator(PoseShiftValidator):
    """Limit root-relative link motion of the measured articulated task objects."""

    check: ClassVar[str] = "articulation_link_shift"

    def skip_reason(self, articulation_keys: list[str]) -> str | None:
        reason = super().skip_reason(articulation_keys)
        if reason is not None:
            return reason
        return None if articulation_keys else "no articulated task objects selected"

    def validate(self, data: SettledBatch) -> list[PlacementValidatorReport]:
        return self._validate_poses(data.env_ids, data.initial_link_poses, data.final_link_poses)


def default_post_physics_validators() -> dict[str, dict]:
    """Default acceptance checks for settled placements."""
    validators = (VelocityValidator(), PoseShiftValidator(), ArticulationLinkShiftValidator())
    return {validator.check: validator.configuration() for validator in validators}


def build_post_physics_validators(
    configurations: dict[str, dict], articulation_keys: list[str], *, log_progress: bool = False
) -> list[PostPhysicsPlacementValidator]:
    """Construct configured checks, optionally printing their settings and skip reasons."""
    from hydra.utils import instantiate

    validators = []
    for name, configuration in configurations.items():
        validator = instantiate(configuration)
        assert isinstance(validator, PostPhysicsPlacementValidator), f"'{name}' must be a PostPhysicsPlacementValidator"
        assert name == validator.check, f"'{name}' must match validator name '{validator.check}'"
        if log_progress:
            reason = validator.skip_reason(articulation_keys)
            status = f"SKIPPED: {reason}" if reason is not None else "ENABLED: required to pass"
            print(f"[placement] {name}: {status}; {validator.configuration()}")
        validators.append(validator)
    assert any(
        validator.skip_reason(articulation_keys) is None for validator in validators
    ), "Enable at least one applicable post-physics validator"
    return validators


@dataclass
class PlacementOutcome:
    """Solver and post-physics verdicts for one sampled candidate."""

    pre_physics: dict[str, bool]
    """Copied source solver verdicts by check name."""
    post_physics: list[PlacementValidatorReport]
    """Configured check outcomes; empty when source solver validation failed."""
    rejection_reason: str | None = None
    """Why the candidate failed; None means all required checks passed."""

    @property
    def passed(self) -> bool:
        """Whether the candidate passed its solver and enabled post-physics checks."""
        return self.rejection_reason is None


def evaluate_settled_batch(
    batch: SettledBatch, validators: Sequence[PostPhysicsPlacementValidator]
) -> dict[int, PlacementOutcome]:
    """Evaluate captured measurements and return an outcome for every source candidate.

    Args:
        batch: Sampled candidates and owned measurements; no live environment is read.
        validators: Configured checks to evaluate for candidates that passed solver validation.

    Returns:
        Outcomes by absolute environment ID, including solver failures and skipped checks.
    """
    from isaaclab_arena.offline_placement.pool_validation import solver_validation_failure

    outcomes = {}
    for env_id, layout in batch.source_layouts.items():
        outcomes[env_id] = PlacementOutcome(
            pre_physics=dict(layout.validation_results.validation_results),
            post_physics=[],
            rejection_reason=solver_validation_failure(layout),
        )
    eligible_env_ids = [env_id for env_id, outcome in outcomes.items() if outcome.passed]
    assert sorted(batch.env_ids) == sorted(
        eligible_env_ids
    ), "Batch environment IDs must match candidates that passed solver checks"
    if not batch.env_ids:
        return outcomes
    for validator in validators:
        reason = validator.skip_reason(list(batch.initial_link_poses))
        if reason is None:
            reports = validator.validate(batch)
            assert all(
                report.passed is not None for report in reports
            ), f"Enabled check '{validator.check}' must return pass/fail"
        else:
            reports = [validator.report(None, reason) for _ in batch.env_ids]
        for env_id, report in zip(batch.env_ids, reports, strict=True):
            outcomes[env_id].post_physics.append(report)
    for env_id in batch.env_ids:
        outcome = outcomes[env_id]
        failures = [f"{report.check}: {report.reason}" for report in outcome.post_physics if report.passed is False]
        if failures:
            outcome.rejection_reason = "; ".join(failures)
    return outcomes
