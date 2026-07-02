# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess smoke test for the runnable Hydra configuration example."""

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


@pytest.mark.with_subprocess
def test_hydra_configuration_example_runs_two_experiments():
    result = run_subprocess(
        [
            TestConstants.python_path,
            "-m",
            "isaaclab_arena_examples.hydra_configuration.run",
            "isaaclab_arena_examples/hydra_configuration/hydra_example_suite.yaml",
            "--viz",
            "none",
            "rollout.num_steps=1",
        ],
        capture_output=True,
    )

    assert result is not None
    assert result.stdout.index("Running job variations_demo") < result.stdout.index(
        "Running job baseline_no_variations"
    )
    assert "[hydra-example] completed jobs=['variations_demo', 'baseline_no_variations']" in result.stdout
