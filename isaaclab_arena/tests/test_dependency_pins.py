# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard that Arena's explicit dependency pins stay in step with the Isaac Lab submodule.

The torch stack is not checked here: isaacsim-core hard-pins it (via the beta-2
wheel), so uv's resolver already fails ``uv lock`` on any mismatch. ``daqp`` is
the one pin Arena chooses freely -- the isaaclab wheel only declares it for ARM,
so the Docker build and the native uv group install it explicitly on x86_64 --
which is why it needs an explicit drift guard against the submodule.
"""

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARENA_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_ISAACLAB_SETUP = _REPO_ROOT / "submodules/IsaacLab/source/isaaclab/setup.py"


def _arena_pin(package: str) -> str:
    """Return the version Arena pins ``package`` to in the isaacsim dependency group."""
    groups = tomllib.loads(_ARENA_PYPROJECT.read_text())["dependency-groups"]["isaacsim"]
    matches = [re.fullmatch(rf"{package}==(.+)", requirement) for requirement in groups]
    versions = [match.group(1) for match in matches if match]
    assert len(versions) == 1, f"expected exactly one {package} pin in the isaacsim group, found {versions}"
    return versions[0]


def _isaaclab_pin(package: str) -> str:
    """Return the version the Isaac Lab submodule's setup.py pins ``package`` to."""
    match = re.search(rf"{package}==([\d.]+)", _ISAACLAB_SETUP.read_text())
    assert match is not None, f"no {package} pin found in {_ISAACLAB_SETUP}"
    return match.group(1)


def test_daqp_pin_matches_isaaclab_submodule():
    assert _arena_pin("daqp") == _isaaclab_pin("daqp")
