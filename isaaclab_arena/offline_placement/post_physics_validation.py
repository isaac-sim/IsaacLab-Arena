# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configurable checks of recorded poses after physics."""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_validation import PlacementValidator, PlacementValidatorReport

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


@dataclass
class PostPhysicsState:
    """Measured poses for N environments, with B links per articulation."""

    env: ManagerBasedEnv
    """Initialized simulation environment."""
    env_ids: list[int]
    """Absolute IDs of candidates that passed required solver checks."""
    initial_poses: dict[str, torch.Tensor]
    """Environment-local initial root poses (N, 7), xyz/xyzw, by scene key."""
    final_poses: dict[str, torch.Tensor]
    """Environment-local final root poses (N, 7), xyz/xyzw, by scene key."""
    initial_links: dict[str, torch.Tensor]
    """Initial root-relative link poses (N, B, 7), xyz/xyzw, by articulation key."""
    final_links: dict[str, torch.Tensor]
    """Final root-relative link poses (N, B, 7), xyz/xyzw, by articulation key."""


@dataclass
class PostPhysicsPlacementValidator(PlacementValidator):
    """A configured check returning one report per candidate environment."""

    stage: ClassVar[str] = "post_physics"
    enabled: bool = True
    """Whether this check must pass for applicable candidates."""

    @abstractmethod
    def validate(self, data: PostPhysicsState) -> list[PlacementValidatorReport]:
        """Return one report per candidate environment, in env_ids order."""

    def configuration(self) -> dict:
        """Return the implementation path and effective settings."""
        return {"_target_": f"{type(self).__module__}.{type(self).__qualname__}", **asdict(self)}

    def skip_reason(self, env: ManagerBasedEnv) -> str | None:
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
    """Require every recorded body to remain below the final velocity limits."""

    check: ClassVar[str] = "physics_settled"
    lin_vel_thresh: float = PhysicsSettleParams.lin_vel_thresh
    """Maximum final linear speed, in m/s."""
    ang_vel_thresh: float = PhysicsSettleParams.ang_vel_thresh
    """Maximum final angular speed, in rad/s."""

    def __post_init__(self) -> None:
        assert (
            math.isfinite(self.lin_vel_thresh) and self.lin_vel_thresh >= 0
        ), "lin_vel_thresh must be finite and non-negative"
        assert (
            math.isfinite(self.ang_vel_thresh) and self.ang_vel_thresh >= 0
        ), "ang_vel_thresh must be finite and non-negative"

    def validate(self, data: PostPhysicsState) -> list[PlacementValidatorReport]:
        from isaaclab_arena.utils.physics_settle import are_all_objects_settled_per_env

        settled = are_all_objects_settled_per_env(
            data.env, data.env_ids, list(data.final_poses), self.lin_vel_thresh, self.ang_vel_thresh
        )
        return [self.report(passed, "" if passed else "objects exceed final velocity limits") for passed in settled]


@dataclass
class PoseShiftValidator(PostPhysicsPlacementValidator):
    """Limit root displacement and rotation from the initial poses."""

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

    def validate(self, data: PostPhysicsState) -> list[PlacementValidatorReport]:
        return self._validate_poses(data.env_ids, data.initial_poses, data.final_poses)

    def _validate_poses(
        self, env_ids: list[int], initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor]
    ) -> list[PlacementValidatorReport]:
        from isaaclab_arena.utils.physics_settle import pose_drift_reason

        reports = []
        for env_id in env_ids:
            reason = ""
            for key, poses in final.items():
                drift = pose_drift_reason(
                    initial[key][env_id], poses[env_id], self.max_translation_m, self.max_rotation_deg
                )
                if drift is not None:
                    reason = f"{key}: {drift}"
                    break
            reports.append(self.report(not bool(reason), reason))
        return reports


@dataclass
class ArticulationLinkShiftValidator(PoseShiftValidator):
    """Limit link motion relative to the root because joint states are not recorded."""

    check: ClassVar[str] = "articulation_link_shift"

    def skip_reason(self, env: ManagerBasedEnv) -> str | None:
        reason = super().skip_reason(env)
        if reason is not None:
            return reason
        return None if env.scene.articulations else "scene has no articulations"

    def validate(self, data: PostPhysicsState) -> list[PlacementValidatorReport]:
        reports = self._validate_poses(data.env_ids, data.initial_links, data.final_links)
        for report in reports:
            if report.passed is False:
                report.reason += "; joint states are not recorded"
        return reports


def default_post_physics_validators() -> dict[str, dict]:
    """Default acceptance checks for recording ordinary solved placements."""
    validators = (VelocityValidator(), PoseShiftValidator(), ArticulationLinkShiftValidator())
    return {validator.check: validator.configuration() for validator in validators}


def build_post_physics_validators(
    configurations: dict[str, dict], env: ManagerBasedEnv
) -> list[PostPhysicsPlacementValidator]:
    """Construct configured checks and print their settings and skip reasons."""
    from hydra.utils import instantiate

    validators = []
    for name, configuration in configurations.items():
        validator = instantiate(configuration)
        assert isinstance(validator, PostPhysicsPlacementValidator), f"'{name}' must be a PostPhysicsPlacementValidator"
        assert name == validator.check, f"'{name}' must match validator name '{validator.check}'"
        reason = validator.skip_reason(env)
        status = f"SKIPPED: {reason}" if reason is not None else "ENABLED: required to pass"
        print(f"[recording] {name}: {status}; {validator.configuration()}")
        validators.append(validator)
    assert any(
        validator.skip_reason(env) is None for validator in validators
    ), "Enable at least one applicable post-physics validator"
    return validators


def validate_post_physics(
    validators: list[PostPhysicsPlacementValidator], data: PostPhysicsState
) -> dict[int, list[PlacementValidatorReport]]:
    """Evaluate every configured check and retain disabled or inapplicable outcomes."""
    results = {env_id: [] for env_id in data.env_ids}
    for validator in validators:
        reason = validator.skip_reason(data.env)
        if reason is None:
            reports = validator.validate(data)
            assert all(
                report.passed is not None for report in reports
            ), f"Enabled check '{validator.check}' must return pass/fail"
        else:
            reports = [validator.report(None, reason) for _ in data.env_ids]
        for env_id, report in zip(data.env_ids, reports, strict=True):
            results[env_id].append(report)
    return results
