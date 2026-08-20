# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.utils.isaaclab_utils.resolve_clone_plan_source_patch import patch_resolve_clone_plan_source


def test_current_isaaclab_clone_query_makes_legacy_patch_a_noop():
    """Current Isaac Lab already resolves nested clone templates through its public query API."""
    try:
        patched = patch_resolve_clone_plan_source()
    except ModuleNotFoundError as exc:
        pytest.fail(f"Arena tried to import Isaac Lab's removed private cloner module: {exc}")

    assert patched is False
