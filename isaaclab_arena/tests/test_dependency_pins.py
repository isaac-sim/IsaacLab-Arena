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

The ``tinyobjloader`` and ``mujoco-usd-converter`` constraints have no submodule
counterpart to drift from: they are transitive Isaac Sim dependencies that
upstream leaves unpinned, so Arena's constraint is itself the source of truth.

TODO(alexmillane, 2026.07.17): Remove after upgrade to Isaac Lab 3.0 GA which
correctly declares its pinned dependencies, obviating the need for pinning in
Arena.
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


def _arena_constraint(package: str) -> str:
    """Return the requirement Arena declares for ``package`` in [tool.uv] constraint-dependencies."""
    constraints = tomllib.loads(_ARENA_PYPROJECT.read_text())["tool"]["uv"]["constraint-dependencies"]
    matches = [constraint for constraint in constraints if re.fullmatch(rf"{package}[=<>!~].*", constraint)]
    assert len(matches) == 1, f"expected exactly one {package} constraint, found {matches}"
    return matches[0]


def _isaaclab_pin(package: str) -> str:
    """Return the version the Isaac Lab submodule's setup.py pins ``package`` to."""
    match = re.search(rf"{package}==([\d.]+)", _ISAACLAB_SETUP.read_text())
    assert match is not None, f"no {package} pin found in {_ISAACLAB_SETUP}"
    return match.group(1)


def test_daqp_pin_matches_isaaclab_submodule():
    assert _arena_pin("daqp") == _isaaclab_pin("daqp")


def test_pin_pink_constraint_matches_isaaclab_submodule():
    assert _arena_constraint("pin-pink") == f"pin-pink=={_isaaclab_pin('pin-pink')}"


def test_pyglet_constraint_matches_isaaclab_submodule():
    """The submodule declares ``pyglet>=2.1.6,<3``; Arena constrains only the upper bound."""
    match = re.search(r"\"pyglet[^\"]*?(<[\d.]+)\"", _ISAACLAB_SETUP.read_text())
    assert match is not None, f"no pyglet upper bound found in {_ISAACLAB_SETUP}"
    assert _arena_constraint("pyglet") == f"pyglet{match.group(1)}"
