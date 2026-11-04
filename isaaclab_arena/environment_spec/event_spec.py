# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serializable Isaac Lab event terms for environment graph specifications."""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventSpec(BaseModel):
    """Configure an additional event without replacing built-in reset events."""

    model_config = ConfigDict(extra="forbid")

    func: str = Field(min_length=1)
    mode: Literal["prestartup", "startup", "reset", "interval"]
    params: dict[str, Any] = Field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require a finite, ordered interval only for interval events."""
        if self.mode == "interval":
            assert self.interval_range_s is not None, "Interval events require interval_range_s"
            low, high = self.interval_range_s
            assert math.isfinite(low) and math.isfinite(high) and 0 <= low <= high, "Invalid event interval"
        else:
            assert self.interval_range_s is None, "Only interval events accept interval_range_s"
        return self


def add_event_terms(env_cfg, events: dict[str, EventSpec]):
    """Append named EventTermCfg entries, rejecting collisions with existing terms."""
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils.string import string_to_callable

    for name, spec in events.items():
        assert name.isidentifier() and not name.startswith("_"), f"Invalid event name: {name}"
        assert not hasattr(env_cfg.events, name), f"Event already exists: {name}"
        values = spec.model_dump()
        values["func"] = string_to_callable(spec.func)
        setattr(env_cfg.events, name, EventTermCfg(**values))
    return env_cfg
