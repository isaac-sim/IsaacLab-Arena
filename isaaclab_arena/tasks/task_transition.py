# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative description of how a task's success changes the env graph state.

A task implements ``TaskBase.success_state_transition`` to return a ``TaskTransition``: the
node the embodiment acts on (if any), plus the effects its success establishes.

Across the task library, every success reduces to two effect kinds:

* ``Relocate`` — a node moves into a spatial relation with another node (pick-and-place,
  sorting). Maps to a binary spatial constraint.
* ``SetState`` — a node's own state changes in place (a door's openness, a pressed button,
  a knob level, or the embodiment opens/closes gripper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Relocate:
    """Success moves ``subject`` into a spatial relation with ``target`` (e.g. ``object in bowl``)."""

    subject: str  # node id whose placement changes (object or embodiment)
    relation: str  # graph spatial-constraint type established (e.g. "in", "on", "at")
    target: str  # node id it ends up related to


@dataclass
class SetState:
    """Success changes ``subject``'s own state in place (e.g. a door's openness), not its location."""

    subject: str  # node id whose state changes
    state: str  # graph constraint type for the state (e.g. "open")
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"openness": 1.0}


Effect = Relocate | SetState


@dataclass
class TaskTransition:
    """What a task acts on and what its success changes in the env graph."""

    subject: str | None = None  # node that the task acts on; None when it changes its own state only (e.g. a GOTO task)
    effects: tuple[Effect, ...] = ()  # Could be both Relocate and SetState; empty if no graph state change

    @property
    def reach_target_on_success(self) -> str | None:
        """The node the embodiment must be able to reach for this task to succeed.

        For a relocation task it is where the object ends up -- the target of the last ``Relocate``
        (e.g. place mug on bowl -> ``bowl``). For an in-place task with no relocation (e.g. open a
        door) it is the ``subject`` the task acts on. ``None`` when neither is defined.
        """
        relocations = [effect for effect in self.effects if isinstance(effect, Relocate)]
        return relocations[-1].target if relocations else self.subject
