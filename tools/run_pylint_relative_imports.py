# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit entrypoint: run pylint with the relative-import plugin only."""

from __future__ import annotations

import subprocess
import sys
from os import environ
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN_MODULE = "pylint_relative_import_plugin"


def main(argv: list[str] | None = None) -> int:
    filenames = list(argv if argv is not None else sys.argv[1:])
    if not filenames:
        return 0

    env = dict(environ)
    tools_dir = str(_REPO_ROOT / "tools")
    env["PYTHONPATH"] = f"{tools_dir}{':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''}"

    pylint_args = [
        sys.executable,
        "-m",
        "pylint",
        "--rcfile",
        str(_REPO_ROOT / ".pylintrc-relative-imports"),
        "--load-plugins",
        _PLUGIN_MODULE,
        "-rn",
        "-sn",
        *filenames,
    ]
    return subprocess.run(pylint_args, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
