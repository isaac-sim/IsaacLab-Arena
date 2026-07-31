# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimReady asset search for agentic environment generation."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from isaaclab_arena.assets.simready_constants import (
    DEFAULT_SIMREADY_SERVICE_URL,
    ISAAC_SIMREADY_GA_S3_URL,
    SIMREADY_PHYSICS_VARIANTS,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
)
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths

MAX_INSPECTED_MATCHES_PER_PHRASE = 5
"""How many hits per object are read to see whether they can be a rigid object, unless more
results than this were asked for. Each read fetches the asset, so a phrase that matches half the
library is not worth chasing to the end."""


class SimReadySourceKind(str, Enum):
    """Configured SimReady search backend."""

    ISAAC_SIM_GA = "isaac-sim-ga"
    S3 = "s3"
    SERVICE = "service"
    CACHE = "cache"
    INDEXED = "indexed"


@dataclass
class SimReadySearchConfig:
    """Configuration for SimReady object lookup."""

    enabled: bool = False
    source: SimReadySourceKind = SimReadySourceKind.ISAAC_SIM_GA
    s3_url: str = ISAAC_SIMREADY_GA_S3_URL
    service_url: str = DEFAULT_SIMREADY_SERVICE_URL
    project_config_path: str | None = None
    indexed_path: str | None = None
    indexed_directory_type: str = "local"
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
    """Prompt-scoped SimReady hits for spec inference."""

    candidates: list[SimReadyObjectCandidate] = field(default_factory=list)

    unmatched_phrases: list[str] = field(default_factory=list)
    """Objects SimReady has nothing usable for, so their spec entry cannot be spawned."""


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _phrase_words(phrase: str) -> list[str]:
    """Split a search phrase into its lowercase words, in order."""
    return [word for word in _NON_ALPHANUMERIC.split(phrase.lower()) if word]


def _phrase_path_filters(phrase: str) -> list[Any]:
    """One path filter per word of the phrase, so a hit only has to match one of them."""
    from simready.search import SearchFilterPathContains

    return [SearchFilterPathContains(word) for word in _phrase_words(phrase)]


def _split_path_into_words(asset_path: str) -> frozenset[str]:
    """Split an asset path into lowercase words, also splitting camelCase.

    SimReady names run words together, as in ``sm_trashCan_wheeled_green_a01_01.usd``, so
    splitting on separators alone would miss ``trash`` and ``can``.
    """
    spaced = _CAMEL_CASE_BOUNDARY.sub(" ", asset_path)
    return frozenset(word for word in _NON_ALPHANUMERIC.split(spaced.lower()) if word)


def _is_word_in_path(word: str, path_words: frozenset[str]) -> bool:
    """True if the word is one of the path words, ignoring a plural on either side."""
    if word in path_words or f"{word}s" in path_words:
        return True
    return word.endswith("s") and word[:-1] in path_words


def _count_matching_words(phrase: str, asset_path: str) -> int:
    """Count how many words of the phrase appear as whole words in the asset path."""
    path_words = _split_path_into_words(asset_path)
    return sum(1 for word in _phrase_words(phrase) if _is_word_in_path(word, path_words))


def _keep_whole_word_matches(matches: list[Any], phrase: str) -> list[Any]:
    """Keep only results named after the object being asked for, best matches first.

    ``SearchFilterPathContains`` matches any substring, so searching for "grey bin" returns every
    cabinet in the library, on the "bin" inside "Cabinets". Whole words alone are not enough
    either, because a grey cabinet still shares "grey" with the phrase. What decides is the last
    word, which is the object itself: the words in front of it only describe it. What survives is
    ordered by the search's own score, where it has one, and then by how much of the phrase the
    asset is named after.
    """
    words = _phrase_words(phrase)
    assert words, "a search phrase needs at least one word"
    object_word = words[-1]
    kept = [match for match in matches if _is_word_in_path(object_word, _split_path_into_words(str(match.asset_path)))]
    # The sort is stable, so hits that tie on both counts keep the order the search gave them.
    kept.sort(
        key=lambda match: (
            getattr(match, "relevance_score", None) or 0.0,
            _count_matching_words(phrase, str(match.asset_path)),
        ),
        reverse=True,
    )
    return kept


def _get_rigid_object_rejection_reason(usd_path: str) -> str | None:
    """Say why a SimReady asset cannot be used as a rigid object, or None if it can.

    Args:
        usd_path: The asset to look at, local or remote.

    Returns:
        A phrase naming the problem, for example "it has no rigid body", or None if the asset is
        usable as a rigid object.
    """
    try:
        rigid_body_paths = read_asset_rigid_body_paths(usd_path, SIMREADY_PHYSICS_VARIANTS)
    except Exception as exc:
        return f"its USD could not be read: {exc}"
    # Only one RigidBodyAPI is allowed on a USD asset.
    if len(rigid_body_paths) == 1:
        return None
    if not rigid_body_paths:
        return "it has no rigid body"
    return f"it has {len(rigid_body_paths)} rigid bodies"


async def _configure_asset_library(config: SimReadySearchConfig, traces: list[str]) -> Any | None:
    try:
        from simready.search import AssetLibrary
    except ImportError:
        traces.append("simready-search is not installed; install with pip install 'isaaclab_arena[simready]'")
        return None

    library = AssetLibrary()
    if config.source in (SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.S3):
        await library.add_s3_source(config.s3_url or ISAAC_SIMREADY_GA_S3_URL)
    elif config.source == SimReadySourceKind.SERVICE:
        library.add_service_source(config.service_url or DEFAULT_SIMREADY_SERVICE_URL)
    elif config.source == SimReadySourceKind.CACHE:
        assert config.project_config_path, "simready cache source requires project_config_path"
        await library.add_cache_source(config.project_config_path)
    elif config.source == SimReadySourceKind.INDEXED:
        assert config.indexed_path, "simready indexed source requires indexed_path"
        await library.add_indexed_source(config.indexed_path, config.indexed_directory_type)
    else:
        traces.append(f"unknown simready source: {config.source}")
        return None
    return library


def _build_simready_object_candidate_from_match(match: Any, phrase: str) -> SimReadyObjectCandidate:
    return SimReadyObjectCandidate(
        search_phrase=phrase,
        usd_path=str(match.asset_path),
        tags=tuple(dict.fromkeys(("sim-ready", *_phrase_words(phrase)))),
        relevance_score=getattr(match, "relevance_score", None),
    )


async def _search_phrase_async(
    library: Any,
    phrase: str,
    *,
    max_results: int,
    traces: list[str],
) -> list[SimReadyObjectCandidate]:
    matches = _keep_whole_word_matches(library.search(include_any=_phrase_path_filters(phrase)), phrase)

    # Reading an asset means fetching it, so only the best few hits are worth checking before we
    # give up on the phrase and let the agent fall back to the Arena asset registry.
    ranked = matches[: max(MAX_INSPECTED_MATCHES_PER_PHRASE, max_results)]
    if len(ranked) < len(matches):
        traces.append(f"simready search checked the best {len(ranked)} of {len(matches)} hits for {phrase!r}")

    candidates: list[SimReadyObjectCandidate] = []
    for match in ranked:
        usd_path = str(match.asset_path)
        # Check if the asset can be used as a rigid object.
        rejection_reason = _get_rigid_object_rejection_reason(usd_path)
        # Assume the asset is usable as a rigid object unless it has a rejection reason.
        if rejection_reason is None:
            candidates.append(_build_simready_object_candidate_from_match(match, phrase))
            if len(candidates) >= max_results:
                break
        else:
            traces.append(f"simready rejected {usd_path} for {phrase!r}: {rejection_reason}")
    return candidates


async def search_simready_objects_async(
    object_phrases: list[str],
    config: SimReadySearchConfig,
    traces: list[str],
) -> SimReadyCandidateCatalogue:
    """Query SimReady for each object phrase.

    Hits that cannot be spawned as a rigid object are turned down, and the next hit is tried in
    their place. A phrase left with nothing is listed as unmatched, so the agent knows to pick
    that object from the Arena asset registry instead.
    """
    phrases = [phrase.strip() for phrase in object_phrases if phrase.strip()]
    if not phrases:
        return SimReadyCandidateCatalogue()

    library = await _configure_asset_library(config, traces)
    if library is None:
        return SimReadyCandidateCatalogue()

    candidates: list[SimReadyObjectCandidate] = []
    unmatched_phrases: list[str] = []
    for phrase in phrases:
        try:
            hits = await _search_phrase_async(
                library,
                phrase,
                max_results=config.max_results_per_object,
                traces=traces,
            )
        except Exception as exc:
            traces.append(f"simready search failed for {phrase!r}: {exc}")
            hits = []
        if hits:
            candidates.extend(hits)
        else:
            traces.append(f"simready search found no usable asset for {phrase!r}")
            unmatched_phrases.append(phrase)

    return SimReadyCandidateCatalogue(candidates=candidates, unmatched_phrases=unmatched_phrases)


def search_simready_objects(
    object_phrases: list[str],
    config: SimReadySearchConfig,
    traces: list[str],
) -> SimReadyCandidateCatalogue:
    """Synchronous wrapper around :func:`search_simready_objects_async`."""
    return asyncio.run(search_simready_objects_async(object_phrases, config, traces))


def simready_search_config_from_cli(
    *,
    enabled: bool,
    source: str,
    s3_url: str | None,
    service_url: str | None,
    project_config_path: str | None,
    indexed_path: str | None,
    max_results_per_object: int,
) -> SimReadySearchConfig:
    """Build :class:`SimReadySearchConfig` from CLI/GUI arguments."""
    return SimReadySearchConfig(
        enabled=enabled,
        source=SimReadySourceKind(source),
        s3_url=s3_url or ISAAC_SIMREADY_GA_S3_URL,
        service_url=service_url or DEFAULT_SIMREADY_SERVICE_URL,
        project_config_path=project_config_path,
        indexed_path=indexed_path,
        max_results_per_object=max_results_per_object,
    )
