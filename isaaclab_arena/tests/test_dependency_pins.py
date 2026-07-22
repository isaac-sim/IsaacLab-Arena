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
_ISAACLAB_PYPROJECT = _REPO_ROOT / "submodules/IsaacLab/pyproject.toml"


def _requirement_name(requirement: str) -> str:
    """Return the normalized package name of a PEP 508 requirement (extras and version dropped)."""
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match is not None, f"cannot parse requirement {requirement!r}"
    return match.group(0).lower().replace("_", "-")


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


def test_isaacsim_source_group_covers_submodule_packages():
    """The isaacsim-source group tracks the submodule's own uv dev project.

    Every isaaclab package upstream's isaaclab-dev project depends on must
    appear in Arena's isaacsim-source dependency group, so a submodule bump
    that adds or renames a source package fails loudly. Arena may declare a
    superset (e.g. isaaclab-mimic and isaaclab-teleop, which the published
    wheel bundles but upstream's dev project omits).
    """
    upstream = tomllib.loads(_ISAACLAB_PYPROJECT.read_text())["project"]["dependencies"]
    upstream_packages = {_requirement_name(req) for req in upstream if _requirement_name(req).startswith("isaaclab")}
    group = tomllib.loads(_ARENA_PYPROJECT.read_text())["dependency-groups"]["isaacsim-source"]
    group_packages = {_requirement_name(req) for req in group if _requirement_name(req).startswith("isaaclab")}
    missing = upstream_packages - group_packages
    assert not missing, f"isaacsim-source group is missing submodule packages: {sorted(missing)}"


def test_isaacsim_source_group_entries_have_path_sources():
    """Every isaaclab package in the isaacsim-source group maps to an editable path source.

    The source must be gated on the isaacsim-source group and point at an
    existing directory in the Isaac Lab submodule; otherwise the package
    silently falls back to the published wheel.
    """
    pyproject = tomllib.loads(_ARENA_PYPROJECT.read_text())
    group = pyproject["dependency-groups"]["isaacsim-source"]
    sources = pyproject["tool"]["uv"]["sources"]
    for package in sorted(_requirement_name(req) for req in group if _requirement_name(req).startswith("isaaclab")):
        assert package in sources, f"no [tool.uv.sources] entry for {package}"
        entries = [entry for entry in sources[package] if entry.get("group") == "isaacsim-source"]
        assert len(entries) == 1, f"expected one isaacsim-source-gated source for {package}, found {entries}"
        (entry,) = entries
        assert entry.get("editable") is True, f"{package} source is not editable"
        path = Path(entry["path"])
        assert path.parts[:3] == (
            "submodules",
            "IsaacLab",
            "source",
        ), f"{package} source path {path} escapes the submodule"
        assert (_REPO_ROOT / path / "setup.py").is_file(), f"{package} source path {path} has no setup.py"
