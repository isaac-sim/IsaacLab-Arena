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
"""How many hits per phrase are inspected for rigid bodies, unless more results than this were
asked for. Inspecting a hit downloads the asset, so a phrase that matches half the library is not
worth following to the end."""


class SimReadySourceKind(str, Enum):
    """Which SimReady backend to search. All of them are remote."""

    ISAAC_SIM_GA = "isaac-sim-ga"
    S3 = "s3"
    SERVICE = "service"


@dataclass
class SimReadySearchConfig:
    """Where to search for SimReady objects and how many hits to keep."""

    source: SimReadySourceKind = SimReadySourceKind.ISAAC_SIM_GA
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


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _phrase_words(phrase: str) -> list[str]:
    """Split a search phrase into its lowercase words, in order."""
    return [word for word in _NON_ALPHANUMERIC.split(phrase.lower()) if word]


def _phrase_path_filters(phrase: str) -> list[Any]:
    """Build one path filter per word of the phrase, so a hit only has to match one word."""
    from simready.search import SearchFilterPathContains

    return [SearchFilterPathContains(word) for word in _phrase_words(phrase)]


def _split_path_into_words(asset_path: str) -> frozenset[str]:
    """Split an asset path into lowercase words, also splitting camelCase.

    SimReady names run words together, as in ``sm_trashCan_wheeled_green_a01_01.usd``. Splitting
    on separators alone would miss ``trash`` and ``can``.
    """
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


def _keep_whole_word_matches(matches: list[Any], phrase: str) -> list[Any]:
    """Keep only the hits named after the object being asked for, best match first.

    ``SearchFilterPathContains`` matches any substring, so "grey bin" returns every cabinet in the
    library, because "Cabinets" contains "bin". Matching whole words is not enough either: a grey
    cabinet still shares "grey" with the phrase. So only the last word decides, because that word
    is the object itself and the words before it merely describe it. The hits that survive are
    ordered by the search's own score, where it has one, and then by how many words of the phrase
    the asset is named after.
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
        A phrase naming the problem, such as "it has no rigid body", or None if the asset is
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


def _configure_asset_library(config: SimReadySearchConfig, traces: list[str]) -> Any | None:
    """Build the SimReady asset library for the configured source, or None if the source is unknown.

    Attaching an S3 source is the only asynchronous call in the whole search, so the event loop
    starts and ends here. Everything after it is synchronous: the query, the ranking, and reading
    an asset to check its rigid bodies.
    """
    # Imported here rather than at module scope only to keep the AWS stack out of the import path
    # of every caller.
    from simready.search import AssetLibrary

    library = AssetLibrary()
    if config.source in (SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.S3):
        asyncio.run(library.add_s3_source(config.s3_url or ISAAC_SIMREADY_GA_S3_URL))
    elif config.source == SimReadySourceKind.SERVICE:
        library.add_service_source(config.service_url or DEFAULT_SIMREADY_SERVICE_URL)
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


def _search_phrase(
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


def search_simready_objects(
    object_phrases: list[str],
    config: SimReadySearchConfig,
    traces: list[str],
) -> SimReadyCandidateCatalogue:
    """Search SimReady for an asset for each object phrase.

    A hit that cannot be spawned as a rigid object is turned down, and the next hit is tried in
    its place. A phrase that runs out of hits is reported as unmatched, so the agent knows to pick
    that object from the Arena asset registry instead.

    Args:
        object_phrases: One phrase per object to look for. Blank phrases are dropped.
        config: Which backend to search and how many hits to keep per object.
        traces: Accumulator for diagnostic lines, extended in place.

    Returns:
        The hits that can be spawned, and the phrases nothing usable was found for.
    """
    phrases = [phrase.strip() for phrase in object_phrases if phrase.strip()]
    if not phrases:
        return SimReadyCandidateCatalogue()

    library = _configure_asset_library(config, traces)
    if library is None:
        return SimReadyCandidateCatalogue()

    candidates: list[SimReadyObjectCandidate] = []
    unmatched_phrases: list[str] = []
    for phrase in phrases:
        try:
            hits = _search_phrase(
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


def simready_search_config_from_cli(
    *,
    source: str,
    s3_url: str | None,
    service_url: str | None,
    max_results_per_object: int,
) -> SimReadySearchConfig:
    """Build a search configuration from CLI or GUI arguments, filling in the default URLs."""
    return SimReadySearchConfig(
        source=SimReadySourceKind(source),
        s3_url=s3_url or ISAAC_SIMREADY_GA_S3_URL,
        service_url=service_url or DEFAULT_SIMREADY_SERVICE_URL,
        max_results_per_object=max_results_per_object,
    )
