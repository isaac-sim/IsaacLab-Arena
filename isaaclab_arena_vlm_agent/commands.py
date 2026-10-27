# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLM command response schemas."""

from pydantic import BaseModel, Field


class Command(BaseModel):
    """Request one command in the adapter-defined layout."""

    values: list[float] = Field(min_length=1)
    """Ordered targets; the adapter defines dimensions, units, and interpretation for the embodiment."""


class CommandChunk(BaseModel):
    """Ordered commands, one per environment step."""

    commands: list[Command] = Field(min_length=1)
    """The action adapter validates each target and enforces the configured chunk length."""
