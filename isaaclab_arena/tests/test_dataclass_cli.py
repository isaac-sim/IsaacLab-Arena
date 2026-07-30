# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the shared dataclass-to-argparse compatibility helpers."""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from isaaclab_arena.cli.dataclass_cli import add_dataclass_cli_args, dataclass_from_cli


class _Mode(str, Enum):
    FAST = "fast"
    SAFE = "safe"


@dataclass
class _ExampleCfg:
    required_value: str
    mode: _Mode = _Mode.FAST
    strategy: Literal["direct", "careful"] = "direct"
    enabled: bool = False
    visible: bool = True
    labels: list[str] = field(default_factory=list)
    threshold: float | None = None
    shared_value: int = 1


def test_generated_arguments_reconstruct_typed_dataclass():
    """Parse representative field types and preserve a shared parser override."""
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument("--shared_value", type=int, default=1)
    add_dataclass_cli_args(parser, _ExampleCfg, excluded_fields={"shared_value"})

    args_cli = parser.parse_args([
        "--required_value",
        "example",
        "--mode",
        "safe",
        "--strategy",
        "careful",
        "--enabled",
        "--no-visible",
        "--labels",
        "left",
        "right",
        "--threshold",
        "1.5",
        "--shared_value",
        "2",
    ])

    assert dataclass_from_cli(_ExampleCfg, args_cli) == _ExampleCfg(
        required_value="example",
        mode=_Mode.SAFE,
        strategy="careful",
        enabled=True,
        visible=False,
        labels=["left", "right"],
        threshold=1.5,
        shared_value=2,
    )
