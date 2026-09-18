# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Combine predicate results while managing child predicate lifecycles."""

from __future__ import annotations

import functools
import torch
from collections.abc import Iterable, Sequence

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate
from isaaclab_arena.tasks.terminations import SuccessMode, combine_success_results


def reset_managed_predicates(
    predicates: Iterable,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Reset each unique managed predicate found through configs or partials."""

    reset_predicate_ids: set[int] = set()
    for predicate in predicates:
        while isinstance(predicate, (TerminationTermCfg, functools.partial)):
            predicate = predicate.func
        if isinstance(predicate, ManagerTermBase) and id(predicate) not in reset_predicate_ids:
            predicate.reset(env_ids)
            reset_predicate_ids.add(id(predicate))


# TODO(cvolk): Revisit CompositePredicate's lifecycle forwarding during the stateful predicate redesign.
# Retain child active masks and selected-environment resets in any replacement.
class CompositePredicate(ConsecutivePredicate):
    """Combine child results, optionally requiring consecutive successful evaluations."""

    def __init__(self, cfg: TerminationTermCfg, env):
        cfg.params.setdefault("consecutive_steps", 1)
        super().__init__(cfg, env)
        self.predicates = cfg.params["predicates"]
        assert self.predicates, "CompositePredicate requires at least one predicate."
        self.results = torch.zeros(
            (len(self.predicates), env.num_envs),
            dtype=torch.bool,
            device=env.device,
        )

    def __call__(
        self,
        env,
        predicates: list[TerminationTermCfg],
        mode: SuccessMode | str = SuccessMode.ALL,
        k: int | None = None,
        consecutive_steps: int = 1,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # These arguments mirror TerminationTermCfg.params for manager signature validation.
        del predicates, consecutive_steps
        results = []
        for predicate in self.predicates:
            predicate_func = predicate.func
            while isinstance(predicate_func, functools.partial):
                predicate_func = predicate_func.func
            parameters = dict(predicate.params)
            if isinstance(predicate_func, ConsecutivePredicate):
                parameters["active_mask"] = active_mask
            results.append(predicate.func(env, **parameters))
        self.results = torch.stack(results, dim=0)
        passed = combine_success_results(self.results, mode=mode, k=k)
        return self._update_consecutive_and_get_completion_mask(passed, active_mask=active_mask)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset managed children and cached results for selected environments."""
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.results[:, ids] = False
        reset_managed_predicates(self.predicates, env_ids)
