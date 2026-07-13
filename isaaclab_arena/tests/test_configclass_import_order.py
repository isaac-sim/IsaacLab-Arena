# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the Isaac Lab wheel configclass import-order shadowing.

With the public Isaac Lab wheel, importing ``isaaclab.envs.mimic_env_cfg`` (or
``isaaclab_newton``) before ``from isaaclab.utils import configclass`` binds the
``configclass`` *submodule* instead of the decorator, so ``@configclass`` fails.
Arena imports the decorator directly from its defining module to stay
order-safe; this test pins that behaviour with a representative manager cfg.
"""

import subprocess
import sys


def test_arena_imports_survive_configclass_shadowing():
    code = (
        "from isaaclab.envs.mimic_env_cfg import MimicEnvCfg; "
        "import isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg as _m; "
        "import isaaclab_arena.recording.progress_terms as _p; "
        "import isaaclab_arena.variations.light_direction_variation as _v; "
        "print('IMPORT_OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Arena configclass import failed:\n{result.stderr}"
    assert "IMPORT_OK" in result.stdout
