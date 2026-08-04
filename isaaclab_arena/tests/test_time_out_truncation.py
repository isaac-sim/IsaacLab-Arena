# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard that every task cfg marks its time-out term as a truncation.

``TerminationTermCfg.time_out`` defaults to ``False``, which makes
``TerminationManager`` report episode-length expiry as ``terminated`` instead of
``truncated``.

The sweep imports every task module, which pulls in Isaac Lab cfg modules, so it runs
in a child process: doing it in the pytest process would trip the duplicate
``ArticulationCfg`` problem that ``test_collection_import_hygiene`` describes and break
every robot-building test that runs afterwards.
"""

import json
import subprocess

from isaaclab_arena.tests.utils.constants import TestConstants

# Packages whose task cfgs must mark their time-out term as a truncation.
_TASK_PACKAGES = ("isaaclab_arena.tasks", "isaaclab_arena_examples.external_environments")

_CHILD_SCRIPT = f"""
import dataclasses, importlib, json, pkgutil

from isaaclab.managers import TerminationTermCfg

terms = {{}}
for package_name in {_TASK_PACKAGES!r}:
    package = importlib.import_module(package_name)
    module_names = [n for _, n, _ in pkgutil.walk_packages(package.__path__, prefix=package_name + ".")]
    for module_name in [package_name, *module_names]:
        module = importlib.import_module(module_name)
        for cls in vars(module).values():
            # Only classes defined in this module, so a cfg imported elsewhere is not double-counted.
            if not isinstance(cls, type) or getattr(cls, "__module__", None) != module_name:
                continue
            # dataclasses.fields() includes inherited fields, unlike __annotations__.
            if not dataclasses.is_dataclass(cls):
                continue
            if not any(f.name == "time_out" for f in dataclasses.fields(cls)):
                continue
            term = cls().time_out
            if isinstance(term, TerminationTermCfg):
                terms[module_name + "." + cls.__name__] = term.time_out
print("TERMS_JSON=" + json.dumps(terms))
"""


def test_all_task_cfgs_mark_time_out_as_truncation():
    """Walks the task packages rather than listing cfgs, so a new task that forgets the
    flag fails here without anyone remembering to update this test."""
    result = subprocess.run(
        [TestConstants.python_path, "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"collecting time_out terms failed:\n{result.stderr}"
    json_lines = [line for line in result.stdout.splitlines() if line.startswith("TERMS_JSON=")]
    assert len(json_lines) == 1, f"marker line not found in child output:\n{result.stdout}"
    terms = json.loads(json_lines[0].removeprefix("TERMS_JSON="))

    assert terms, f"No time_out terms discovered in {_TASK_PACKAGES}; the sweep is not testing anything."
    offenders = sorted(name for name, is_time_out in terms.items() if is_time_out is not True)
    assert not offenders, f"Termination cfgs declaring a time_out term without time_out=True: {offenders}"
