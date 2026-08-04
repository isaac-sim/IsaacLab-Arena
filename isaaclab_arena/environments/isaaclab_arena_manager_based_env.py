# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.variation_recorder import VariationRecorder

# Physics-step ceiling for settling a clutter pile, measured on piles of 8 to 80 objects:
# 8 settle by ~250 steps, 40 by ~1000, 80 by ~2000. Settling returns as soon as the poses go
# quiet, so these bound the wait rather than always being spent.
_MIN_CLUTTER_SETTLE_STEPS = 400
_CLUTTER_SETTLE_STEPS_PER_MEMBER = 30


def external_policy_termination(env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
    """Return environments whose policy requested non-timeout episode termination."""
    return env.external_policy_termination_buf


class IsaacLabArenaManagerBasedRLEnv(ManagerBasedRLEnv):
    """Arena extension to ManagerBasedRLEnv that adds additional Arena-specific functionality."""

    cfg: IsaacLabArenaManagerBasedRLEnvCfg

    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        variation_recorder: VariationRecorder | None = None,
        **kwargs,
    ):
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(
            num_envs=cfg.scene.num_envs, device=cfg.sim.device
        )
        self._variation_recorder = variation_recorder
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        # Per-env count of completed episodes; advanced in ``_reset_idx``.
        self._episode_counts: dict[int, int] = {}
        # The initial reset touches every env before any episode has run; skip it.
        self._first_reset = True
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._external_policy_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._settle_clutter_layouts()

    def _settle_clutter_layouts(self) -> None:
        """Replace pooled clutter drop poses with the poses the pile settles into.

        A pour only describes where objects are released. Writing those poses at reset would
        start every episode with the pile in mid-air, so the pool is settled once here and
        each layout keeps its resting arrangement for every reset that draws it.
        """
        import math

        from isaaclab_arena.relations.clutter_groups import get_clutter_groups
        from isaaclab_arena.relations.clutter_validation import ClutterSettleParams
        from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
        from isaaclab_arena.relations.placement_events import get_placement_pool
        from isaaclab_arena.relations.placement_pool_validation import validate_pool_layouts

        if not self.cfg.settle_clutter_on_build:
            return
        placement_pool = get_placement_pool(self)
        if placement_pool is None:
            return
        groups = get_clutter_groups(placement_pool.objects)
        if not groups:
            return

        # The default settle budget is sized for a couple of objects nudging into place; a pile
        # needs orders of magnitude more, growing with how deep it stacks. Settling returns as
        # soon as the poses go quiet, so a generous ceiling costs nothing on an easy pile.
        member_count = sum(len(group.members) for group in groups)
        physics_steps = max(_MIN_CLUTTER_SETTLE_STEPS, _CLUTTER_SETTLE_STEPS_PER_MEMBER * member_count)
        settle_params = PhysicsSettleParams(num_steps=math.ceil(physics_steps / self.cfg.decimation))

        validate_pool_layouts(
            self,
            placement_pool=placement_pool,
            settle_params=settle_params,
            capture_settled_poses=True,
            pose_settle_params=ClutterSettleParams(),
        )

    @property
    def variation_recorder(self) -> VariationRecorder | None:
        """The recorder of variation samples, or ``None`` if the env was not built with one."""
        if self._variation_recorder is None:
            print(
                "[WARNING] variation_recorder is None; no variation samples were recorded. "
                "Build the env through ArenaEnvBuilder to record variations."
            )
        return self._variation_recorder

    @property
    def object_initial_rest_pose_recorder(self) -> ObjectInitialRestPoseRecorder:
        """The recorder of initial object rest poses. Used when object_settled predicate is enabled by task progress tracking."""
        return self._object_initial_rest_pose_recorder

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    def load_managers(self) -> None:
        super().load_managers()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the index of the current episode in ``env_id``."""
        return self._episode_counts.get(env_id, 0)

    @property
    def external_policy_termination_buf(self) -> torch.Tensor:
        """Boolean per-environment policy termination requests for the current step."""
        return self._external_policy_termination_buf

    def request_external_policy_termination(self, termination_mask: torch.Tensor) -> None:
        """Request non-timeout episode termination for environments selected by ``termination_mask``."""
        assert isinstance(termination_mask, torch.Tensor), "termination_mask must be a torch.Tensor"
        assert termination_mask.dtype == torch.bool, "termination_mask must have dtype torch.bool"
        assert termination_mask.shape == (
            self.num_envs,
        ), f"termination_mask must have shape ({self.num_envs},), got {tuple(termination_mask.shape)}"
        self._external_policy_termination_buf |= termination_mask.to(device=self.device)

    def _advance_episode_indices(self, env_ids: Sequence[int]) -> None:
        """Advance the per-env episode counter for each episode in ``env_ids``."""
        for env_id in env_ids:
            env_id = int(env_id)
            self._episode_counts[env_id] = self._episode_counts.get(env_id, 0) + 1

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # The initial reset touches every env before any episode has run; nothing to record or count.
        if self._first_reset:
            self._first_reset = False
            super()._reset_idx(env_ids)
            self.episode_recorder_manager.record_post_reset(env_ids)
            return
        # Runs recorder before super() so the just-finished episode is still intact.
        self.episode_recorder_manager.record_pre_reset(env_ids)
        # Preserve the signal through recording, then clear it before the next episode starts.
        self._external_policy_termination_buf[env_ids] = False
        # Advance before super() so reset-mode variation draws are tagged with the episode they begin.
        self._advance_episode_indices(env_ids)
        super()._reset_idx(env_ids)
        self.episode_recorder_manager.record_post_reset(env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
