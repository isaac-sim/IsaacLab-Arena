# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for handling OSMO workflow IDs."""

import re

# OSMO workflow IDs are limited to this character set. It is also the set that is safe to
# splice into a shell command unquoted (no whitespace, quotes, or shell metacharacters).
_WORKFLOW_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def is_valid_workflow_id(workflow_id: str) -> bool:
    """Return whether a string is a well-formed OSMO workflow ID.

    Callers that interpolate an ID into a generated shell script rely on this: an ID that
    matches contains no whitespace or shell metacharacters, so it cannot word-split or
    inject when spliced into a command unquoted.
    """
    return _WORKFLOW_ID_PATTERN.fullmatch(workflow_id) is not None
