# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena git commit and branch that produced an Experiment's results."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

UNKNOWN_GIT_STAMP_VALUE = "unknown"
"""Value recorded when the git commit or branch is not known."""

ARENA_GIT_SHA_ENV_VAR = "ARENA_GIT_SHA"
"""Environment variable holding the Arena git commit SHA; the Arena Docker image sets it."""

ARENA_GIT_BRANCH_ENV_VAR = "ARENA_GIT_BRANCH"
"""Environment variable holding the Arena git branch name; the Arena Docker image sets it."""


@dataclass(frozen=True)
class GitStamp:
    """Arena git commit and branch that produced an Experiment's results."""

    git_sha: str
    """Git commit SHA, suffixed with ``-dirty`` for uncommitted changes, or ``"unknown"``."""

    git_branch: str
    """Git branch name, or ``"unknown"``."""

    def __post_init__(self) -> None:
        for git_stamp_field in fields(self):
            value = getattr(self, git_stamp_field.name)
            assert (
                isinstance(value, str) and value and "\n" not in value and "\r" not in value
            ), f"git stamp {git_stamp_field.name} must be a non-empty single-line string, got {value!r}"

    @classmethod
    def from_environment(cls) -> GitStamp:
        """Return the git stamp from ``ARENA_GIT_SHA`` and ``ARENA_GIT_BRANCH``, ``"unknown"`` when unset or empty."""
        return cls(
            git_sha=os.environ.get(ARENA_GIT_SHA_ENV_VAR) or UNKNOWN_GIT_STAMP_VALUE,
            git_branch=os.environ.get(ARENA_GIT_BRANCH_ENV_VAR) or UNKNOWN_GIT_STAMP_VALUE,
        )

    @classmethod
    def from_dict(cls, git_stamp: Mapping[str, str]) -> GitStamp:
        """Return the git stamp stored as a JSON object with exactly ``git_sha`` and ``git_branch``."""
        expected_keys = {git_stamp_field.name for git_stamp_field in fields(cls)}
        assert (
            set(git_stamp) == expected_keys
        ), f"git stamp must have exactly the keys {sorted(expected_keys)}, got {sorted(git_stamp)}"
        return cls(**git_stamp)

    def to_dict(self) -> dict[str, str]:
        """Return the git stamp as a JSON-compatible object with ``git_sha`` and ``git_branch``."""
        return asdict(self)
