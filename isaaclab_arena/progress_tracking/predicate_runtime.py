# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare predicate occurrences and share evaluations within one tracker update."""

from __future__ import annotations

import copy
import functools
import torch

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.tasks.predicates.stateful_predicate import (
    Predicate,
    PreparedPredicate,
    StatefulPredicate,
    StatefulPredicateCfg,
    is_predicate,
    predicate_description,
)


class PredicateFactory:
    """Prepare fresh stateful occurrences and resolve Isaac Lab predicate configurations."""

    def __init__(self, num_envs: int, device, env=None):
        self.num_envs = num_envs
        self.device = device
        self.env = env

    def prepare(self, predicate: Predicate) -> PreparedPredicate:
        """Create stateful occurrences while preserving ordinary callable identity."""
        assert is_predicate(predicate), "Expected a callable, TerminationTermCfg, or StatefulPredicateCfg."
        if isinstance(predicate, StatefulPredicateCfg):
            runtime = predicate.create_runtime(self)
            assert isinstance(runtime, StatefulPredicate), "create_runtime must return a StatefulPredicate."
            return runtime
        if not isinstance(predicate, TerminationTermCfg):
            return predicate
        assert self.env is not None, "An environment is required to initialize a configured progress predicate."
        predicate_cfg = copy.deepcopy(predicate)
        self._initialize_parameters(predicate_cfg)
        return functools.partial(predicate_cfg.func, **predicate_cfg.params)

    def _initialize_parameters(self, value) -> None:
        """Resolve scene references and construct nested Isaac Lab terms before their parents."""
        # TaskSuccessTerm constructs the tracker before env.termination_manager exists.
        # Isaac Lab therefore cannot initialize these nested configurations for us.
        if isinstance(value, TerminationTermCfg):
            self._initialize_parameters(value.params)
            if isinstance(value.func, type):
                value.func = value.func(value, self.env)
            assert callable(value.func), "Predicate configs must resolve to a callable."
        elif isinstance(value, SceneEntityCfg):
            value.resolve(self.env.scene)
        elif isinstance(value, dict):
            for parameter in value.values():
                self._initialize_parameters(parameter)
        elif isinstance(value, (list, tuple)):
            for parameter in value:
                self._initialize_parameters(parameter)


class PredicateEvaluation:
    """Cache predicate results for one control step, including nested predicate evaluations."""

    def __init__(self, env, num_envs: int, device):
        self.env = env
        self.num_envs = num_envs
        self.device = device
        self._results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def evaluate(self, predicate: PreparedPredicate, active_envs: torch.Tensor) -> torch.Tensor:
        """Evaluate each stateful occurrence at most once per active environment.

        Ordinary callables evaluate the full batch once, keyed by callable identity.
        An empty active mask reads cached stateful results without activating the runtime.
        """
        predicate_key = id(predicate)
        if predicate_key not in self._results:
            self._results[predicate_key] = (
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
        cached_result, evaluated_envs = self._results[predicate_key]
        stateful = isinstance(predicate, StatefulPredicate)
        requested_envs = active_envs if stateful else torch.ones_like(active_envs)
        pending_envs = requested_envs & ~evaluated_envs
        if bool(pending_envs.any().item()):
            result = predicate.evaluate(self, pending_envs) if stateful else predicate(self.env)
            result = torch.as_tensor(result, dtype=torch.bool, device=self.device)
            assert result.shape == (self.num_envs,), (
                f"Predicate {predicate_description(predicate)} returned shape {tuple(result.shape)};"
                f" expected ({self.num_envs},)"
            )
            cached_result = torch.where(pending_envs, result, cached_result)
            self._results[predicate_key] = (cached_result, evaluated_envs | pending_envs)
        return cached_result
