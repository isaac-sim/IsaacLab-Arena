# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable reports from the existing graph and agent-ready catalogue checks."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.catalogues import (
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.spec_validation import collect_agent_ready_validation_trace
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec


def validate_authoring_spec(data: dict[str, Any]) -> dict[str, Any]:
    """Report existing schema and catalogue violations without constructing an environment.

    Args:
        data: Environment graph mapping to check.

    Returns:
        A JSON-compatible report. Parameter semantics and physical feasibility still
        require normal construction and simulation; this adds no validation rules.
    """
    issues = []
    try:
        spec = ArenaEnvGraphSpec.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors(include_context=False, include_input=False):
            path = "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in error["loc"])
            issues.append(_issue("schema_error", path, error["msg"]))
    else:
        messages = collect_agent_ready_validation_trace(
            spec, build_asset_catalogue(), build_task_catalogue(), build_relation_catalogue()
        )
        issues = [_issue("catalogue_error", "/", message) for message in messages]
    return {
        "schema_version": 1,
        "valid": not issues,
        "validation_scope": "schema_and_catalogue",
        "issues": issues,
    }


def _issue(code: str, path: str, message: str) -> dict[str, Any]:
    return {"code": code, "path": path, "message": message}
