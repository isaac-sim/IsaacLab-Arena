# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asserting that every leftover argparse ``unknown`` token is a Hydra override."""

from __future__ import annotations

import argparse

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
