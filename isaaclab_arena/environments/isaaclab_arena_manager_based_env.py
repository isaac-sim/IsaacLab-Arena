# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
from isaaclab_arena.relations.clutter_groups import get_clutter_groups
from isaaclab_arena.relations.clutter_pour import region_for_support, resting_extents
from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, check_resting_poses
from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_events import get_placement_pool
from isaaclab_arena.relations.placement_pool_validation import CAPTURED_OBJECTS_SETTLED, validate_pool_layouts
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.variation_recorder import VariationRecorder

_MIN_CLUTTER_SETTLE_STEPS = 400
"""Physics steps allowed for any clutter pile to settle, however small."""

_CLUTTER_SETTLE_STEPS_PER_MEMBER = 30
"""Further physics steps allowed per pile member. Settling stops as soon as the poses go
quiet, so this bounds the wait rather than being spent."""


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
        self._settle_clutter_layouts()

    def _settle_clutter_layouts(self) -> None:
        """Replace pooled clutter drop poses with the poses the pile settles into.

        A pour only describes where objects are released. Writing those poses at reset would
        start every episode with the pile in mid-air, so the pool is settled once here and
        each layout keeps its resting arrangement for every reset that draws it.
        """
        if not self.cfg.settle_clutter_on_build:
            return
        placement_pool = get_placement_pool(self)
        if placement_pool is None:
            return
        groups = get_clutter_groups(placement_pool.objects)
        if not groups:
            return

        # A pile needs far longer to settle than the default budget allows, and longer the
        # deeper it stacks, so size the budget by how many members it holds.
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
        # Settling steps physics, which a reset cannot do, so an exhausted queue must rewind
        # rather than solve fresh layouts this pass would never see. Without this, every reset
        # past the cached set writes the poses the pile was released from and the pile falls.
        placement_pool.recycle_layouts = True
        self._reject_spilled_clutter_layouts(placement_pool, groups)

    def _reject_spilled_clutter_layouts(self, placement_pool, groups) -> None:
        """Drop cached layouts whose pile did not stay on its support.

        Whether a pour spills is only knowable after settling it, so the pool is filtered
        afterwards rather than constrained up front. An env keeps its rejects when too few
        layouts survive, since a spilled pile still beats having nothing to draw.
        """
        per_env_bboxes = build_per_env_bounding_boxes(
            placement_pool.objects, self.num_envs
        ).get_bounding_boxes_for_all_envs()
        params = ClutterSettleParams(containment_margin_m=self.cfg.clutter_containment_margin_m)

        def keep(env_id: int, layout) -> bool:
            # A pile that never went quiet still holds its release poses, which sit above the
            # support and inside the region, so containment alone cannot tell it from a settled
            # one. Drop it rather than replay a falling pile at every reset.
            if layout.validation_results.validation_results.get(CAPTURED_OBJECTS_SETTLED) is False:
                return False

            bboxes = per_env_bboxes[env_id]
            for group in groups:
                support = group.support
                # Judge against the whole support, not the shrunk region the pile was poured
                # into: a tight pour is meant to relax outward as it settles. The region is
                # resolved the same way the pour resolved it, so the two cannot disagree.
                region = region_for_support(support, layout, bboxes)
                members = [member for member in group.members if member in layout.positions]
                positions = torch.tensor([layout.positions[member] for member in members], dtype=torch.float32)
                # Judge each member by the box it came to rest in, so one whose origin sits over
                # the support while its body hangs off the edge is not accepted.
                extents = [resting_extents(member, layout, bboxes[member]) for member in members]
                if positions.numel() and not check_resting_poses(positions, region, params, extents).ok:
                    return False
            return True

        # Recycling rewinds past the cursor, so the layout construction already consumed comes
        # back and has to be judged with the rest.
        kept, rejected, starved = placement_pool.retain_layouts(keep, include_consumed=True)
        # An env with nothing left has only spilled piles to draw, and would run every episode
        # against objects lying beside the support rather than on it -- wrong data that looks
        # like data. One layout in the whole pool staying put is a low bar to clear, so failing
        # it means the pour cannot fit this pile on this support, not that physics was unlucky.
        assert not starved, (
            f"Every cached clutter layout spilled in env(s) {starved}. The pile does not fit the "
            "support it is poured onto: use fewer or smaller members, raise clutter_containment_"
            f"margin_m (currently {self.cfg.clutter_containment_margin_m}), or lower the group's "
            "spread so members are released further from the edge."
        )
        print(f"[clutter] rejected {rejected} spilled layout(s); {kept} cached")

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
            return
        # Runs recorder before super() so the just-finished episode is still intact.
        self.episode_recorder_manager.record_pre_reset(env_ids)
        # Advance before super() so reset-mode variation draws are tagged with the episode they begin.
        self._advance_episode_indices(env_ids)
        super()._reset_idx(env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
