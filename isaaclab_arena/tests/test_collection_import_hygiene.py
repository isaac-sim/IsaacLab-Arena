# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard that no test module imports Isaac Lab cfg modules at collection time.

Pytest imports every test module during collection, before the ``SimulationApp``
launches. If any of them (directly or transitively) executes Isaac Lab cfg
modules such as ``isaaclab.assets``, ``AppLauncher``'s ``sys.modules``
delete/restore hack later mints a duplicate ``ArticulationCfg`` during Kit
startup, and every robot-building test collected afterwards fails
``isinstance`` checks with "Unknown asset config type" (see IsaacLab #6514).
Keep such imports inside the inner functions run via
``run_simulation_app_function``.

TODO(alexmillane, 2026-07-15): Remove this test once Isaac Lab #6514 is fixed.
"""

import json
import subprocess

from isaaclab_arena.tests.utils.constants import TestConstants

_FORBIDDEN_PREFIXES = ("isaaclab.assets", "isaaclab.scene", "isaaclab_assets", "isaaclab_tasks")

_CHILD_SCRIPT = f"""
import glob, importlib, json, sys

offenders = {{}}
for path in sorted(glob.glob(sys.argv[1] + "/test_*.py")):
    module_name = "isaaclab_arena.tests." + path.rsplit("/", 1)[-1][:-3]
    before = set(sys.modules)
    importlib.import_module(module_name)
    leaked = sorted(n for n in set(sys.modules) - before if n.startswith({_FORBIDDEN_PREFIXES!r}))
    if leaked:
        offenders[module_name] = leaked
print("OFFENDERS_JSON=" + json.dumps(offenders))
"""


def test_no_isaaclab_cfg_imports_at_collection_time():
    result = subprocess.run(
        [TestConstants.python_path, "-c", _CHILD_SCRIPT, TestConstants.test_dir],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"importing the test modules failed:\n{result.stderr}"
    json_lines = [line for line in result.stdout.splitlines() if line.startswith("OFFENDERS_JSON=")]
    assert len(json_lines) == 1, f"marker line not found in child output:\n{result.stdout}"
    offenders = json.loads(json_lines[0].removeprefix("OFFENDERS_JSON="))
    assert not offenders, (
        "These test modules import Isaac Lab cfg modules at collection (module) time, which duplicates "
        f"ArticulationCfg once the SimulationApp launches: {offenders}. "
        "Defer the imports into the inner function run via run_simulation_app_function."
    )
