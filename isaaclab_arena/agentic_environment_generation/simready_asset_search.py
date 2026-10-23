# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SimReady asset search for agentic environment generation."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.simready_constants import (
    DEFAULT_SIMREADY_SERVICE_URL,
    ISAAC_SIMREADY_GA_S3_URL,
    SIMREADY_PHYSICS_VARIANTS,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
)
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths

if TYPE_CHECKING:
    from simready.search import AssetLibrary

_logger = logging.getLogger(__name__)

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
    source_url: str | None = None

    @property
    def registry_name(self) -> str:
        return SIMREADY_USD_OBJECT_REGISTRY_NAME


@dataclass
class SimReadyCandidateCatalogue:
    """The SimReady hits found for one prompt, handed to spec inference."""

    candidates: list[SimReadyObjectCandidate] = field(default_factory=list)

    unmatched_phrases: list[str] = field(default_factory=list)
    """Objects the search found no usable asset for, so they are left out of the spec."""


def search_simready_objects(object_phrases: list[str], config: SimReadySearchConfig) -> SimReadyCandidateCatalogue:
    """Search SimReady for an asset for each object phrase. A hit that cannot be spawned as a rigid object is turned down, and the next hit is tried in
    its place.

    Args:
        object_phrases: One phrase per object to look for. Blank phrases are dropped.
        config: Which backend to search and how many hits to keep per object.

    Returns:
        The hits that can be spawned, and the phrases nothing usable was found for.
    """
    phrases = [phrase.strip() for phrase in object_phrases if phrase.strip()]
    if not phrases:
        return SimReadyCandidateCatalogue()

    candidates = []
    seen = set()
    for source in dict.fromkeys(config.sources):
        try:
            result = _search_simready_source(phrases, config, source)
        except Exception as exc:
            _logger.warning("simready backend %s failed: %s", source, exc)
            continue
        for candidate in result.candidates:
            key = (candidate.search_phrase, candidate.usd_path)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    matched = {candidate.search_phrase for candidate in candidates}
    return SimReadyCandidateCatalogue(
        candidates=candidates,
        unmatched_phrases=[phrase for phrase in phrases if phrase not in matched],
    )


def _search_simready_source(
    phrases: list[str], config: SimReadySearchConfig, source: SimReadySourceKind
) -> SimReadyCandidateCatalogue:
    """Search one source, retaining the configured number of hits per phrase."""
    library = (
        None
        if source == SimReadySourceKind.CENTRAL
        else _configure_asset_library(source, s3_url=config.s3_url, service_url=config.service_url)
    )
    if library is None and source != SimReadySourceKind.CENTRAL:
        return SimReadyCandidateCatalogue()

    candidates: list[SimReadyObjectCandidate] = []
    unmatched_phrases: list[str] = []
    for phrase in phrases:
        try:
            if source == SimReadySourceKind.CENTRAL:
                from isaaclab_arena.agentic_environment_generation.simready_central import search_central_objects

                hits = search_central_objects(phrase, max_results=config.max_results_per_object)
            else:
                # Query the configured asset library through the simready-search API.
                hits = _search_phrase(library, phrase, max_results=config.max_results_per_object)
        except Exception as exc:
            _logger.warning("simready search failed for %r: %s", phrase, exc)
            hits = []
        if hits:
            candidates.extend(hits)
        else:
            _logger.info("simready search found no usable asset for %r", phrase)
            unmatched_phrases.append(phrase)

    return SimReadyCandidateCatalogue(candidates=candidates, unmatched_phrases=unmatched_phrases)


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


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _configure_asset_library(source: SimReadySourceKind, *, s3_url: str, service_url: str) -> Any | None:
    """Build the SimReady asset library for the configured source, or None if the source is unknown."""
    if source not in tuple(SimReadySourceKind):
        _logger.error("unknown simready source: %s", source)
        return None

    # Imported here rather than at module scope only to keep the AWS stack out of the import path
    # of every caller.
    from simready.search import AssetLibrary

    library = AssetLibrary()
    if source in (SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.S3):
        asyncio.run(library.add_s3_source(s3_url or ISAAC_SIMREADY_GA_S3_URL))
    else:
        library.add_service_source(service_url or DEFAULT_SIMREADY_SERVICE_URL)
    return library


def _search_phrase(library: AssetLibrary, phrase: str, *, max_results: int) -> list[SimReadyObjectCandidate]:
    matches = _keep_whole_word_matches(library.search(include_any=_phrase_path_filters(phrase)), phrase)

    # Inspect a few more hits than are wanted back, so a rejected one can be replaced rather than
    # costing the phrase a result. Past that, give up and let the agent fall back to the Arena
    # asset registry: every further hit inspected is another asset downloaded.
    ranked = matches[: max(MIN_INSPECTED_MATCHES_PER_PHRASE, max_results)]
    if len(ranked) < len(matches):
        _logger.info("simready search checked the best %d of %d hits for %r", len(ranked), len(matches), phrase)

    candidates: list[SimReadyObjectCandidate] = []
    for match in ranked:
        usd_path = str(match.asset_path)
        is_valid, rejection_reason = _is_valid_isaaclab_rigidbody(usd_path)
        if is_valid:
            candidates.append(_build_simready_object_candidate_from_match(match, phrase))
            if len(candidates) >= max_results:
                break
        else:
            _logger.info("simready rejected %s for %r: %s", usd_path, phrase, rejection_reason)
    return candidates


def _phrase_path_filters(phrase: str) -> list[Any]:
    """Build one path filter per word of the phrase, so a hit only has to match one word."""
    from simready.search import SearchFilterPathContains

    return [SearchFilterPathContains(word) for word in _phrase_words(phrase)]


def _keep_whole_word_matches(matches: list[Any], phrase: str) -> list[Any]:
    """Keep only the hits named after the object being asked for, best match first."""
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


def _is_valid_isaaclab_rigidbody(usd_path: str) -> tuple[bool, str]:
    """Say whether a SimReady asset can be spawned as an Isaac Lab rigid object.

    Args:
        usd_path: The asset to look at, local or remote.

    Returns:
        Whether the asset is usable, and a phrase naming the problem when it is not, such as
        "it has no rigid body". The phrase is empty when the asset is usable.
    """
    try:
        rigid_body_paths = read_asset_rigid_body_paths(usd_path, SIMREADY_PHYSICS_VARIANTS)
    except Exception as exc:
        return False, f"its USD could not be read: {exc}"
    # Only one RigidBodyAPI is allowed on a USD asset.
    if len(rigid_body_paths) == 1:
        return True, ""
    if not rigid_body_paths:
        return False, "it has no rigid body"
    return False, f"it has {len(rigid_body_paths)} rigid bodies"


def _build_simready_object_candidate_from_match(match: Any, phrase: str) -> SimReadyObjectCandidate:
    return SimReadyObjectCandidate(
        search_phrase=phrase,
        usd_path=str(match.asset_path),
        tags=tuple(dict.fromkeys(("sim-ready", *_phrase_words(phrase)))),
        relevance_score=getattr(match, "relevance_score", None),
    )


def _phrase_words(phrase: str) -> list[str]:
    """Split a search phrase into its lowercase words, in order."""
    return [word for word in _NON_ALPHANUMERIC.split(phrase.lower()) if word]


def _split_path_into_words(asset_path: str) -> frozenset[str]:
    """Split an asset path into lowercase words, also splitting camelCase."""
    # e.g. trashcan_wheeled_green_a01_01.usd -> trashcan wheeled green a01 01
    spaced = _CAMEL_CASE_BOUNDARY.sub(" ", asset_path)
    return frozenset(word for word in _NON_ALPHANUMERIC.split(spaced.lower()) if word)


def _is_word_in_path(word: str, path_words: frozenset[str]) -> bool:
    """True if the word is one of the path words. A trailing "s" on either side is ignored."""
    if word in path_words or f"{word}s" in path_words:
        return True
    return word.endswith("s") and word[:-1] in path_words


def _count_matching_words(phrase: str, asset_path: str) -> int:
    """Count how many words of the phrase appear as whole words in the asset path."""
    path_words = _split_path_into_words(asset_path)
    return sum(1 for word in _phrase_words(phrase) if _is_word_in_path(word, path_words))
