# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from isaaclab_arena.agentic_environment_generation.agent_utils import (
    apply_strict_constraints,
    build_strict_schema,
    ping,
)

# ---------------------------------------------------------------------------
# build_strict_schema / apply_strict_constraints
# ---------------------------------------------------------------------------


class _ToyChild(BaseModel):
    name: str
    optional_value: int | None = None


class _ToyParent(BaseModel):
    title: str
    child: _ToyChild
    children: list[_ToyChild] = []


class TestBuildStrictSchema:
    def test_apply_strict_constraints_is_idempotent(self):
        # Safe to call multiple times — the second pass must be a no-op.
        schema = build_strict_schema(_ToyParent)
        snapshot = json.dumps(schema, sort_keys=True)
        apply_strict_constraints(schema)
        assert json.dumps(schema, sort_keys=True) == snapshot


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_error_paths(self):
        class FakeAuthError(Exception):
            pass

        client = MagicMock()
        # SDK exceptions propagate unchanged.
        client.chat.completions.create.side_effect = FakeAuthError("invalid api key")
        with pytest.raises(FakeAuthError, match="invalid api key"):
            ping(client, "m")

        # Same client, different failure shape: 200 OK with no choices.
        client.chat.completions.create.side_effect = None
        resp = MagicMock()
        resp.choices = []
        client.chat.completions.create.return_value = resp
        with pytest.raises(AssertionError, match="no choices") as exc_info:
            ping(client, "guardrailed-model")
        # Model name surfaces in the message — most-grepped field when
        # triaging a CI ping failure.
        assert "'guardrailed-model'" in str(exc_info.value)
