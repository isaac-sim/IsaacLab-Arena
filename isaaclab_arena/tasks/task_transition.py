# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative description of how a task's success changes the env graph state.

A task implements ``TaskBase.success_state_transition`` to return a ``TaskTransition``: the
node the embodiment acts on (if any), plus the effects its success establishes.

Across the task library, every success reduces to two effect kinds:

* ``Relocate`` — a node moves into a spatial relation with another node (pick-and-place,
  sorting, or the embodiment navigating to a location). Maps to a binary spatial constraint.
* ``SetState`` — a node's own state changes in place (a door's openness, a pressed button,
  a knob level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass()
class Relocate:
    """Success moves ``subject`` into a spatial relation with ``target`` (e.g. ``object in bowl``).
    """

    subject: str  # node id whose placement changes (object or embodiment)
    relation: str  # graph spatial-constraint type established (e.g. "in", "on", "at")
    target: str  # node id it ends up related to


@dataclass()
class SetState:
    """Success changes ``subject``'s own state in place (e.g. a door's openness), not its location."""

    subject: str  # node id whose state changes
    state: str  # graph constraint type for the state (e.g. "open")
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"openness": 1.0}


Effect = Relocate | SetState


@dataclass()
class TaskTransition:
    """How a task's success changes the env-graph state."""

    subject: str | None = None  # node that the task acts on; None when it
    # only changes its own state (e.g. a GOTO task, whose outcome is an embodiment Relocate effect)
    effects: tuple[Effect, ...] = ()  # Could be both Relocate and SetState; empty if no graph state change

    # Note: every state gets a REACH constraint for the embodiment to head for.
    # The two cases differ because a state plays two different roles:
    # - A non-terminal state i+1 is the initial state of task i+1 -> the robot should be poised to begin that next task -> reach its subject (the
    # thing it's about to pick up / act on).
    # - The terminal state has no "next task" -> the robot should be where the last task left it -> that's reach_target_on_success.
    @property
    def reach_target_on_success(self) -> str | None:
        """End of the task's success: the last relocation's target, else ``subject`` (may be None)."""
        relocations = [effect for effect in self.effects if isinstance(effect, Relocate)]
        return relocations[-1].target if relocations else self.subject
