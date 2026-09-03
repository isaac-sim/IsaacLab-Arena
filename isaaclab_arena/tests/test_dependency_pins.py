# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard Arena's source dependency wiring against Isaac Lab submodule drift.

Arena repeats a small set of upstream uv overrides because nested uv projects
do not contribute their tool configuration to the root lock. The remaining
Arena-only constraints cover transitive Isaac Sim dependencies whose upstream
requirements are intentionally broad.
"""

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARENA_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_ISAACLAB_PYPROJECT = _REPO_ROOT / "submodules/IsaacLab/pyproject.toml"


def _requirement_name(requirement: str) -> str:
    """Return the normalized package name of a PEP 508 requirement (extras and version dropped)."""
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match is not None, f"cannot parse requirement {requirement!r}"
    return match.group(0).lower().replace("_", "-")


def _arena_constraint(package: str) -> str:
    """Return the requirement Arena declares for ``package`` in [tool.uv] constraint-dependencies."""
    constraints = tomllib.loads(_ARENA_PYPROJECT.read_text())["tool"]["uv"]["constraint-dependencies"]
    matches = [constraint for constraint in constraints if re.fullmatch(rf"{package}[=<>!~].*", constraint)]
    assert len(matches) == 1, f"expected exactly one {package} constraint, found {matches}"
    return matches[0]


def _isaaclab_pin(package: str) -> str:
    """Return the version the Isaac Lab development project pins ``package`` to."""
    dependencies = tomllib.loads(_ISAACLAB_PYPROJECT.read_text())["project"]["dependencies"]
    requirements = [requirement for requirement in dependencies if _requirement_name(requirement) == package]
    assert len(requirements) == 1, f"expected one {package} requirement in {_ISAACLAB_PYPROJECT}, found {requirements}"
    match = re.search(r"==([\d.]+)", requirements[0])
    assert match is not None, f"no exact {package} pin found in {_ISAACLAB_PYPROJECT}"
    return match.group(1)


def test_isaaclab_uv_overrides_match_submodule():
    """Arena mirrors the source checkout's backend and torch overrides exactly."""
    packages = {
        "mujoco",
        "mujoco-warp",
        "newton",
        "newton-usd-schemas",
        "torch",
        "torchaudio",
        "torchvision",
        "typing-extensions",
    }
    arena_overrides = tomllib.loads(_ARENA_PYPROJECT.read_text())["tool"]["uv"]["override-dependencies"]
    isaaclab_overrides = tomllib.loads(_ISAACLAB_PYPROJECT.read_text())["tool"]["uv"]["override-dependencies"]
    arena_by_package = {_requirement_name(requirement): requirement for requirement in arena_overrides}
    isaaclab_by_package = {_requirement_name(requirement): requirement for requirement in isaaclab_overrides}
    assert {package: arena_by_package[package] for package in packages} == {
        package: isaaclab_by_package[package] for package in packages
    }
    assert arena_by_package["warp-lang"] == f"warp-lang=={_isaaclab_pin('warp-lang')}"


def test_pin_pink_constraint_matches_isaaclab_submodule():
    assert _arena_constraint("pin-pink") == f"pin-pink=={_isaaclab_pin('pin-pink')}"


def test_pyglet_constraint_matches_isaaclab_submodule():
    """The submodule declares ``pyglet>=2.1.6,<3``; Arena constrains only the upper bound."""
    dependencies = tomllib.loads(_ISAACLAB_PYPROJECT.read_text())["project"]["dependencies"]
    requirements = [requirement for requirement in dependencies if _requirement_name(requirement) == "pyglet"]
    assert len(requirements) == 1, f"expected one pyglet requirement in {_ISAACLAB_PYPROJECT}, found {requirements}"
    match = re.search(r"(<[\d.]+)", requirements[0])
    assert match is not None, f"no pyglet upper bound found in {_ISAACLAB_PYPROJECT}"
    assert _arena_constraint("pyglet") == f"pyglet{match.group(1)}"


def test_isaaclab_from_source_group_covers_submodule_dev_project():
    """The source flavor installs Isaac Lab's dev project and all of its workspace members."""
    upstream = tomllib.loads(_ISAACLAB_PYPROJECT.read_text())["project"]["dependencies"]
    expected_packages = {
        _requirement_name(requirement)
        for requirement in upstream
        if _requirement_name(requirement).startswith("isaaclab")
    }
    expected_packages.add("isaaclab-dev")
    group = tomllib.loads(_ARENA_PYPROJECT.read_text())["dependency-groups"]["isaaclab-from-source"]
    group_packages = {_requirement_name(requirement) for requirement in group if isinstance(requirement, str)}
    missing_packages = expected_packages - group_packages
    assert not missing_packages, f"isaaclab-from-source group is missing packages: {sorted(missing_packages)}"


def test_isaaclab_from_source_packages_have_path_sources():
    """Every package in Isaac Lab's dev project maps to an editable submodule source.

    The source must be gated on the isaaclab-from-source group and point at an
    existing directory in the Isaac Lab submodule; otherwise the resolver can
    silently select a package-index distribution.
    """
    pyproject = tomllib.loads(_ARENA_PYPROJECT.read_text())
    upstream = tomllib.loads(_ISAACLAB_PYPROJECT.read_text())["project"]["dependencies"]
    upstream_packages = {
        _requirement_name(requirement)
        for requirement in upstream
        if _requirement_name(requirement).startswith("isaaclab")
    }
    sources = pyproject["tool"]["uv"]["sources"]
    submodule_root = (_REPO_ROOT / "submodules/IsaacLab").resolve()
    for package in sorted(upstream_packages | {"isaaclab-dev"}):
        assert package in sources, f"no [tool.uv.sources] entry for {package}"
        entries = [entry for entry in sources[package] if entry.get("group") == "isaaclab-from-source"]
        assert len(entries) == 1, f"expected one isaaclab-from-source-gated source for {package}, found {entries}"
        (entry,) = entries
        assert entry.get("editable") is True, f"{package} source is not editable"
        path = (_REPO_ROOT / entry["path"]).resolve()
        assert path.is_relative_to(submodule_root), f"{package} source path {path} escapes the submodule"
        assert (path / "pyproject.toml").is_file(), f"{package} source path {path} has no pyproject.toml"
