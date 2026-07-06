# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asserting that every leftover argparse ``unknown`` token is a Hydra override."""

from __future__ import annotations

import argparse
import json
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser


def assert_hydra_overrides(args: list[str], parser: argparse.ArgumentParser) -> None:
    """Assert args are all Hydra overrides.

    Args:
        args: The arguments to assert are all Hydra overrides.
        parser: The parser the args came from; used to format the error.
    """
    overrides_parser = OverridesParser.create()
    # Parse token-by-token so the error message can name the offending tokens.
    bad = []
    for arg in args:
        try:
            overrides_parser.parse_overrides([arg])
        except Exception:  # noqa: BLE001 -- Hydra raises its own parse exceptions
            bad.append(arg)
    if bad:
        parser.error(f"Unrecognized arguments: {' '.join(bad)}")


def hydra_overrides_from_nested_dict(values: dict[str, Any]) -> list[str]:
    """Flatten nested values into dotted Hydra override strings."""
    overrides: list[str] = []
    stack: list[tuple[str, object]] = [("", values)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, dict):
            for key, value in reversed(tuple(node.items())):
                assert key, "Hydra override keys must be non-empty"
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                stack.append((child_prefix, value))
            continue

        assert prefix, "Hydra override path must be non-empty"
        overrides.append(f"{prefix}={json.dumps(node, separators=(',', ':'))}")
    return overrides
