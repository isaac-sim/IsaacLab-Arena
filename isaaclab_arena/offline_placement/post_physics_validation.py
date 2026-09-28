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
    """Initial task-object link poses relative to the root (N, B, 7), xyz/xyzw, by scene key."""
    final_links: dict[str, torch.Tensor]
    """Final task-object link poses relative to the root (N, B, 7), xyz/xyzw, by scene key."""


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

    def validate(self, data: PostPhysicsState) -> list[PlacementValidatorReport]:
        from isaaclab_arena.utils.physics_settle import are_all_objects_settled_per_env

        settled = are_all_objects_settled_per_env(
            data.env, data.env_ids, list(data.final_poses), self.lin_vel_thresh, self.ang_vel_thresh
        )
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
    """Limit root-relative link motion of the measured articulated task objects."""

    check: ClassVar[str] = "articulation_link_shift"

    def skip_reason(self, articulation_keys: list[str]) -> str | None:
        reason = super().skip_reason(articulation_keys)
        if reason is not None:
            return reason
        return None if articulation_keys else "no articulated task objects selected"

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
    configurations: dict[str, dict], articulation_keys: list[str]
) -> list[PostPhysicsPlacementValidator]:
    """Construct configured checks and print their settings and skip reasons."""
    from hydra.utils import instantiate

    validators = []
    for name, configuration in configurations.items():
        validator = instantiate(configuration)
        assert isinstance(validator, PostPhysicsPlacementValidator), f"'{name}' must be a PostPhysicsPlacementValidator"
        assert name == validator.check, f"'{name}' must match validator name '{validator.check}'"
        reason = validator.skip_reason(articulation_keys)
        status = f"SKIPPED: {reason}" if reason is not None else "ENABLED: required to pass"
        print(f"[recording] {name}: {status}; {validator.configuration()}")
        validators.append(validator)
    assert any(
        validator.skip_reason(articulation_keys) is None for validator in validators
    ), "Enable at least one applicable post-physics validator"
    return validators


def validate_post_physics(
    validators: list[PostPhysicsPlacementValidator], data: PostPhysicsState
) -> dict[int, list[PlacementValidatorReport]]:
    """Evaluate every configured check and retain disabled or inapplicable outcomes."""
    results = {env_id: [] for env_id in data.env_ids}
    for validator in validators:
        reason = validator.skip_reason(list(data.initial_links))
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


def articulation_link_poses_in_root_frame(
    env: ManagerBasedEnv, articulation_keys: list[str]
) -> dict[str, torch.Tensor]:
    """Return selected link-to-root poses (N, B, 7), with N environments and B links per articulation."""
    import torch

    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

    poses = {}
    for key in articulation_keys:
        body = env.scene.articulations[key]
        links = body.data.body_link_pose_w.torch
        root = env.arena_world.get_pose_w(key)[:, None, :].expand_as(links)
        position = quat_apply_inverse(root[..., 3:], links[..., :3] - root[..., :3])
        rotation = quat_mul(quat_conjugate(root[..., 3:]), links[..., 3:])
        poses[key] = torch.cat((position, rotation), dim=-1)
    return poses
