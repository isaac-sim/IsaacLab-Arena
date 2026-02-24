# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy package exports.

This module avoids eager importing of all policy implementations because some
policies trigger asset registration side effects during import. Training code
that only needs lightweight helpers (for example `get_agent_cfg`) should not
pull in remote asset registry dependencies.
"""

from __future__ import annotations

from importlib import import_module

from .action_chunking import ActionChunkingState

__all__ = [
    "ActionChunkingState",
    "ActionChunkingClientSidePolicy",
    "ReplayActionPolicy",
    "RslRlActionPolicy",
    "ZeroActionPolicy",
]


_LAZY_IMPORTS = {
    "ActionChunkingClientSidePolicy": "isaaclab_arena.policy.action_chunking_client",
    "ReplayActionPolicy": "isaaclab_arena.policy.replay_action_policy",
    "RslRlActionPolicy": "isaaclab_arena.policy.rsl_rl_action_policy",
    "ZeroActionPolicy": "isaaclab_arena.policy.zero_action_policy",
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_IMPORTS[name])
    return getattr(module, name)
