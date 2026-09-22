# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared configclass overrides."""

from dataclasses import field

import pytest
from isaaclab.utils.configclass import configclass

from isaaclab_arena.hydra.config_override import apply_config_override, dotlist_to_override, nested_override


@configclass
class _ChildCfg:
    count: int = 1
    labels: list[str] = field(default_factory=lambda: ["default"])


@configclass
class _RootCfg:
    child: _ChildCfg = field(default_factory=_ChildCfg)
    runs: dict[str, _ChildCfg] = field(default_factory=lambda: {"baseline": _ChildCfg()})


def test_apply_config_override_updates_nested_configclass_and_mapping():
    cfg = _RootCfg()
    child = cfg.child
    baseline = cfg.runs["baseline"]

    result = apply_config_override(
        cfg,
        {
            "child": {"count": 3},
            "runs": {"baseline": {"labels": ["one", "two"]}},
        },
    )

    assert result is cfg
    assert cfg.child.count == 3
    assert cfg.runs["baseline"].labels == ["one", "two"]
    assert cfg.child is child
    assert cfg.runs["baseline"] is baseline


def test_apply_config_override_is_atomic_on_unknown_path():
    cfg = _RootCfg()

    with pytest.raises(ValueError, match="Key not found"):
        apply_config_override(cfg, {"child": {"count": 3}, "missing": True})

    assert cfg == _RootCfg()


def test_dotlist_to_override_parses_typed_values():
    override = dotlist_to_override([
        "child.count=4",
        "child.labels=[first,second]",
    ])

    assert override == {"child": {"count": 4, "labels": ["first", "second"]}}


def test_nested_override_expands_dotted_keys():
    override = nested_override({
        "child.count": 5,
        "runs": {"baseline.labels": ["updated"]},
    })

    assert override == {
        "child": {"count": 5},
        "runs": {"baseline": {"labels": ["updated"]}},
    }


@pytest.mark.parametrize("operator", ["+", "++", "~"])
def test_dotlist_to_override_rejects_hydra_operators(operator):
    with pytest.raises(ValueError, match="operators are not supported"):
        dotlist_to_override([f"{operator}child.count=4"])
