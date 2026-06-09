# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asserting that every leftover argparse ``unknown`` token is a Hydra override."""

from __future__ import annotations

import argparse
import re

# Hydra override token shapes we accept on the CLI
# See: https://hydra.cc/docs/advanced/override_grammar/basic/
_HYDRA_KEY = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_HYDRA_OVERRIDE_RE = re.compile(rf"^(?:~{_HYDRA_KEY}(?:=.*)?|(?:\+{{1,2}})?{_HYDRA_KEY}=.*)$")


def assert_hydra_overrides(args: list[str], parser: argparse.ArgumentParser) -> None:
    """Assert args are all Hydra overrides.

    Args:
        args: The arguments to assert are all Hydra overrides.
        parser: The parser the args came from; used to format the error.
    """
    bad = [arg for arg in args if not _HYDRA_OVERRIDE_RE.match(arg)]
    if bad:
        parser.error(f"Unrecognized arguments: {' '.join(bad)}")
