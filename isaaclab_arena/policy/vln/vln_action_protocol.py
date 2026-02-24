# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-export VlnVelocityActionProtocol from the Arena core package.

The VLN velocity-command ActionProtocol is defined in
``isaaclab_arena.remote_policy.action_protocol`` alongside the other
protocol types (ChunkingActionProtocol, etc.).  This module re-exports
it for convenience.
"""

from isaaclab_arena.remote_policy.action_protocol import (  # noqa: F401
    ActionMode,
    VlnVelocityActionProtocol,
)
