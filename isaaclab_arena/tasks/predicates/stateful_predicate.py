# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarations and lifecycle contracts for predicates with per-environment state."""

from __future__ import annotations

import functools
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.managers import TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab_arena.progress_tracking.predicate_runtime import PredicateEvaluation, PredicateFactory


class StatefulPredicateCfg(ABC):
    """Declare a predicate whose runtime state belongs to one configured occurrence."""

    @abstractmethod
    def create_runtime(self, factory: PredicateFactory) -> StatefulPredicate:
        """Create a fresh runtime, preparing any children through factory."""

    def describe(self) -> str:
        """Return the predicate name and configured parameters for progress reports."""
        return repr(self)


class StatefulPredicate(ABC):
    """Own one occurrence's episode state, including the state of any child predicates.

    CompletionCriteriaRunner selects active environments and resets runtime roots.
    Implementations evaluate children through PredicateEvaluation and reset owned children.
    """

    requires_step_index = False
    """Whether consecutive control-step indices are required by this runtime or any owned child."""

    def __init__(self, description: str):
        self.description = description

    @abstractmethod
    def evaluate(self, evaluation: PredicateEvaluation, active_envs: torch.Tensor) -> torch.Tensor:
        """Return one Boolean per environment, updating state only for active_envs."""

    @abstractmethod
    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        """Clear episode state for the selected environments, including owned children."""

    @property
    def diagnostics(self) -> Callable | StatefulPredicate:
        """Return the predicate supplying diagnostic data to tracker consumers."""
        return self


Predicate = Callable | TerminationTermCfg | StatefulPredicateCfg
PreparedPredicate = Callable | StatefulPredicate


def is_predicate(value) -> bool:
    """Return whether value is a declaration or an initialized instantaneous callable."""
    if isinstance(value, StatefulPredicate):
        return False
    return isinstance(value, (TerminationTermCfg, StatefulPredicateCfg)) or (
        callable(value) and not isinstance(value, type)
    )


def predicate_description(predicate: Predicate | PreparedPredicate) -> str:
    """Describe a declaration or prepared predicate for progress reports."""
    if isinstance(predicate, StatefulPredicateCfg):
        return predicate.describe()
    if isinstance(predicate, StatefulPredicate):
        return predicate.description
    if isinstance(predicate, TerminationTermCfg):
        predicate = functools.partial(predicate.func, **predicate.params)
    if isinstance(predicate, functools.partial):
        function, arguments, parameters = predicate.func, predicate.args, (predicate.keywords or {})
    else:
        function, arguments, parameters = predicate, (), {}
    name = getattr(function, "__name__", type(function).__name__)
    parts = [repr(argument) for argument in arguments]
    parts += [f"{key}={value!r}" for key, value in parameters.items() if isinstance(value, (str, int, float, bool))]
    return f"{name}({', '.join(parts)})" if parts else name


def predicate_diagnostics(predicate: PreparedPredicate) -> Callable | StatefulPredicate:
    """Return diagnostic data from the underlying predicate."""
    if isinstance(predicate, StatefulPredicate):
        return predicate.diagnostics
    while isinstance(predicate, functools.partial):
        predicate = predicate.func
    return predicate
