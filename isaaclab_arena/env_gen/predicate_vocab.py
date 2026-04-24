"""Shared predicate name handling for env_gen.

GoalSpec predicates appear in two distinct pipeline stages, each consumed
by a separate module:

- `predicate_map.py` — success_conditions → TerminationTermCfg
  (runtime: "did the robot succeed?")
- `predicate_to_relation.py` — goal_relations → Arena Relation
  (scene setup: "where should objects be placed?")

The two modules handle overlapping predicate names (e.g. `on_top`,
`left_of`) but produce different outputs. This module holds only logic
that must stay in sync across both — currently just name normalization.
"""

from __future__ import annotations


def normalize_predicate(predicate: str) -> str:
    """Strip the optional `object_` prefix so callers can match either form."""
    return predicate.removeprefix("object_") if predicate.startswith("object_") else predicate
