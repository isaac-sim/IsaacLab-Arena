# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys


# TODO(xinjie.yao, 2026.03.31): Remove it after policy sever-client is implemented properly in v0.3.
def ensure_groot_deps_in_path(reexec_argv: list[str] | None = None) -> None:
    """Prepend ``GROOT_DEPS_DIR`` to ``PYTHONPATH`` and re-exec the process so
    GR00T dependencies are importable before Isaac Sim's bundled packages.

    The function is guarded by the ``_GROOT_PYTHONPATH_APPLIED`` env-var so it
    only re-execs once.  If ``GROOT_DEPS_DIR`` is not set the call is a no-op.

    Args:
        reexec_argv: The argv list to pass to ``os.execv`` after the Python
            interpreter.  Defaults to ``sys.argv`` (i.e. re-run the current
            script with the same arguments).  Pass
            ``["-m", "pytest"] + sys.argv[1:]`` when bootstrapping from a
            pytest conftest so the test runner is invoked correctly.
    """
    deps_dir = os.environ.get("GROOT_DEPS_DIR")
    if not deps_dir or os.environ.get("_GROOT_PYTHONPATH_APPLIED") == "1":
        return

    os.environ["PYTHONPATH"] = deps_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["_GROOT_PYTHONPATH_APPLIED"] = "1"

    if reexec_argv is None:
        reexec_argv = sys.argv
    os.execv(sys.executable, [sys.executable] + reexec_argv)
