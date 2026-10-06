# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the Dexsuite Kuka Allegro lift Arena example."""

import ast
import re
import shutil
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


@pytest.mark.with_subprocess
def test_dexsuite_lift_published_checkpoint(tmp_path: Path) -> None:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    published_checkpoint = get_published_pretrained_checkpoint(
        "rsl_rl", "Isaac-Lift-KukaAllegro", "newtonmjwarp", "none"
    )
    assert published_checkpoint is not None, "Isaac Lab's published Dexsuite lift checkpoint is unavailable"

    checkpoint_path = tmp_path / "Isaac-Lift-KukaAllegro.pt"
    shutil.copy2(published_checkpoint, checkpoint_path)
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    shutil.copy2(
        Path(TestConstants.repo_root) / "isaaclab_arena_examples/policy/dexsuite_lift_agent.yaml",
        params_dir / "agent.yaml",
    )

    result = run_subprocess(
        [
            TestConstants.python_path,
            f"{TestConstants.evaluation_dir}/policy_runner.py",
            "--policy_type",
            "rsl_rl",
            "--num_episodes",
            "1",
            "--num_envs",
            "1",
            "--checkpoint_path",
            str(checkpoint_path),
            "dexsuite_lift",
        ],
        capture_output=True,
    )
    assert result is not None
    output = result.stdout + result.stderr
    metrics_matches = re.findall(r"Metrics: (\{[^\n]+\})", output)
    assert metrics_matches, f"Evaluation did not report metrics:\n{output}"
    metrics = ast.literal_eval(metrics_matches[-1])
    assert metrics["num_episodes"] == 1
