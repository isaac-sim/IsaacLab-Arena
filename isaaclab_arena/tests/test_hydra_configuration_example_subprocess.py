# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess smoke test for the runnable Hydra configuration example."""

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


@pytest.mark.with_subprocess
def test_hydra_configuration_example_runs():
    result = run_subprocess(
        [
            TestConstants.python_path,
            "-m",
            "isaaclab_arena_examples.hydra_configuration.run",
            "--viz",
            "none",
            "rollout.num_steps=1",
        ],
        capture_output=True,
    )

    assert result is not None
    assert "[hydra-example] completed 'maple_table_zero_action'" in result.stdout
