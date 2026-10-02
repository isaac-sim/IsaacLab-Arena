# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for agent-generated environment graph specs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

if TYPE_CHECKING:
    from isaaclab_arena.agentic_environment_generation.catalogues import (
        AssetCatalogue,
        RelationCatalogue,
        TaskCatalogue,
    )


_ASSERTION_FAILED_PREFIX = "Assertion failed, "


def _clean_validation_msg(msg: str) -> str:
    """Strip pydantic's assertion wrapper from validator error text."""
    if msg.startswith(_ASSERTION_FAILED_PREFIX):
        return msg[len(_ASSERTION_FAILED_PREFIX) :]
    return msg


def format_validation_error(exc: ValidationError) -> list[str]:
    """Flatten a Pydantic ``ValidationError`` into human-readable trace lines."""
    lines: list[str] = []
    for err in exc.errors():
        msg = _clean_validation_msg(err["msg"])
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"{loc}: {msg}" if loc else msg)
    return lines


def collect_agent_ready_validation_trace(
    spec: ArenaEnvGraphSpec,
    asset_catalog: AssetCatalogue,
    task_catalog: TaskCatalogue,
    relation_catalog: RelationCatalogue,
) -> list[str]:
    """Return spec violations against the exact catalogues exposed to the agent."""
    from isaaclab_arena.agentic_environment_generation.semantic_validation import collect_semantic_validation_issues

    return [
        issue.message
        for issue in collect_semantic_validation_issues(
            spec.model_dump(mode="json"), asset_catalog, task_catalog, relation_catalog
        )
    ]
