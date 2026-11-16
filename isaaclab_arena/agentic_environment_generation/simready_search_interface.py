# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared types and protocol for SimReady search backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from isaaclab_arena.assets.simready_constants import (
    DEFAULT_SIMREADY_SERVICE_URL,
    ISAAC_SIMREADY_GA_S3_URL,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
)

MIN_INSPECTED_MATCHES_PER_PHRASE = 5
"""How many hits per phrase to inspect for rigid bodies, unless more results than this were asked for."""


class SimReadySourceKind(str, Enum):
    """Which SimReady backend to search."""

    ISAAC_SIM_GA = "isaac-sim-ga"
    S3 = "s3"
    SERVICE = "service"
    CENTRAL = "simready-central"


@dataclass
class SimReadySearchConfig:
    """Where to search for SimReady objects and how many hits to keep."""

    sources: list[SimReadySourceKind] = field(default_factory=lambda: [SimReadySourceKind.ISAAC_SIM_GA])
    s3_url: str = ISAAC_SIMREADY_GA_S3_URL
    service_url: str = DEFAULT_SIMREADY_SERVICE_URL
    max_results_per_object: int = 1


@dataclass(frozen=True)
class SimReadyObjectCandidate:
    """One SimReady search hit exposed to spec inference."""

    search_phrase: str
    usd_path: str
    tags: tuple[str, ...] = ("sim-ready",)
    relevance_score: float | None = None

    @property
    def registry_name(self) -> str:
        return SIMREADY_USD_OBJECT_REGISTRY_NAME


@dataclass
class SimReadyCandidateCatalogue:
    """The SimReady hits found for one prompt, handed to spec inference."""

    candidates: list[SimReadyObjectCandidate] = field(default_factory=list)

    unmatched_phrases: list[str] = field(default_factory=list)
    """Objects the search found no usable asset for, so they are left out of the spec."""


class SimReadySearchBackend(Protocol):
    """Return validated candidates for object phrases from one backend."""

    def search(self, phrases: list[str], max_results: int) -> list[SimReadyObjectCandidate]:
        """Search each phrase, retaining at most max_results candidates per phrase."""
        ...


def simready_search_config_from_cli(
    sources: list[str],
    s3_url: str | None,
    service_url: str | None,
    max_results_per_object: int,
) -> SimReadySearchConfig:
    """Build a search configuration from CLI or GUI arguments, filling in the default URLs."""
    return SimReadySearchConfig(
        sources=[SimReadySourceKind(source) for source in dict.fromkeys(sources)],
        s3_url=s3_url or ISAAC_SIMREADY_GA_S3_URL,
        service_url=service_url or DEFAULT_SIMREADY_SERVICE_URL,
        max_results_per_object=max_results_per_object,
    )
