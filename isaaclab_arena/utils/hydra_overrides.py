# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pulling Hydra-shaped variation override tokens out of an argparse ``unknown`` list."""

from __future__ import annotations

import argparse
import re

# Hydra override token shapes we accept on the CLI
# See: https://hydra.cc/docs/advanced/override_grammar/basic/
_HYDRA_KEY = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_HYDRA_OVERRIDE_RE = re.compile(rf"^(?:~{_HYDRA_KEY}(?:=.*)?|(?:\+{{1,2}})?{_HYDRA_KEY}=.*)$")


def split_hydra_overrides(unknown: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """Pull Hydra-shaped override tokens out of an argparse ``unknown`` list.

    Any leftover that does not match a Hydra override shape (see
    :data:`_HYDRA_OVERRIDE_RE`) is rejected via ``parser.error``, exiting the
    script with code 2 -- the same behaviour strict :meth:`parse_args` had.

    Args:
        unknown: Second return value of ``parser.parse_known_args()``.
        parser: The parser the unknowns came from; used to format the error.

    Returns:
        The Hydra override tokens, in original order.
    """
    overrides: list[str] = []
    bad: list[str] = []
    for token in unknown:
        if _HYDRA_OVERRIDE_RE.match(token):
            overrides.append(token)
        else:
            bad.append(token)
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)}")
    return overrides
