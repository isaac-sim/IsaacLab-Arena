# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Search GA, S3, and service assets through the simready-search AssetLibrary API."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from isaaclab_arena.agentic_environment_generation.simready_search_interface import (
    MIN_INSPECTED_MATCHES_PER_PHRASE,
    SimReadyObjectCandidate,
    SimReadySourceKind,
)
from isaaclab_arena.assets.simready_constants import (
    DEFAULT_SIMREADY_SERVICE_URL,
    ISAAC_SIMREADY_GA_S3_URL,
    SIMREADY_PHYSICS_VARIANTS,
)
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths

if TYPE_CHECKING:
    from simready.search import AssetLibrary
_logger = logging.getLogger(__name__)


class AssetLibrarySearchBackend:
    """Search a GA, S3, or service source through AssetLibrary."""

    def __init__(self, source: SimReadySourceKind, *, s3_url: str, service_url: str):
        self.library = _create_asset_library_for_source(source, s3_url=s3_url, service_url=service_url)

    def search(self, phrases: list[str], max_results: int) -> list[SimReadyObjectCandidate]:
        """Search each phrase and retain single-rigid-body candidates."""
        if self.library is None:
            return []
        candidates = []
        for phrase in phrases:
            try:
                # Query the simready-search API through its AssetLibrary backend.
                hits = _search_library_rigid_body_candidates(self.library, phrase, max_results=max_results)
                if not hits:
                    _logger.info("simready found no usable asset for %r", phrase)
                candidates.extend(hits)
            except Exception as exc:
                _logger.warning("simready search failed for %r: %s", phrase, exc)
        return candidates


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _create_asset_library_for_source(source: SimReadySourceKind, *, s3_url: str, service_url: str) -> Any | None:
    """Build the SimReady asset library for the configured source, or None if the source is unknown."""
    if source not in (SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.S3, SimReadySourceKind.SERVICE):
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


def _search_library_rigid_body_candidates(
    library: AssetLibrary, phrase: str, *, max_results: int
) -> list[SimReadyObjectCandidate]:
    """Search the library by phrase and retain ranked assets with exactly one rigid body."""
    matches = _filter_and_rank_path_matches(library.search(include_any=_build_path_filters_from_phrase(phrase)), phrase)

    # Inspect a few more hits than are wanted back, so a rejected one can be replaced rather than
    # costing the phrase a result. Past that, give up and let the agent fall back to the Arena
    # asset registry: every further hit inspected is another asset downloaded.
    ranked = matches[: max(MIN_INSPECTED_MATCHES_PER_PHRASE, max_results)]
    if len(ranked) < len(matches):
        _logger.info("simready search checked the best %d of %d hits for %r", len(ranked), len(matches), phrase)

    candidates: list[SimReadyObjectCandidate] = []
    for match in ranked:
        usd_path = str(match.asset_path)
        is_valid, rejection_reason = _has_single_rigid_body(usd_path)
        if is_valid:
            candidates.append(_build_simready_object_candidate_from_match(match, phrase))
            if len(candidates) >= max_results:
                break
        else:
            _logger.info("simready rejected %s for %r: %s", usd_path, phrase, rejection_reason)
    return candidates


def _build_path_filters_from_phrase(phrase: str) -> list[Any]:
    """Build one path filter per word of the phrase, so a hit only has to match one word."""
    from simready.search import SearchFilterPathContains

    return [SearchFilterPathContains(word) for word in _split_phrase_into_words(phrase)]


def _filter_and_rank_path_matches(matches: list[Any], phrase: str) -> list[Any]:
    """Keep only the hits named after the object being asked for, best match first."""
    words = _split_phrase_into_words(phrase)
    assert words, "a search phrase needs at least one word"
    object_word = words[-1]
    kept = [
        match
        for match in matches
        if _path_words_match_singular_or_plural(object_word, _split_path_into_words(str(match.asset_path)))
    ]
    # The sort is stable, so hits that tie on both counts keep the order the search gave them.
    kept.sort(
        key=lambda match: (
            getattr(match, "relevance_score", None) or 0.0,
            _count_phrase_words_in_path(phrase, str(match.asset_path)),
        ),
        reverse=True,
    )
    return kept


def _has_single_rigid_body(usd_path: str) -> tuple[bool, str]:
    """Check that a readable USD contains exactly one rigid body with the SimReady variants selected.

    Args:
        usd_path: The asset to look at, local or remote.

    Returns:
        Whether exactly one rigid body was found, and a reason when the check fails, such as
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
        tags=tuple(dict.fromkeys(("sim-ready", *_split_phrase_into_words(phrase)))),
        relevance_score=getattr(match, "relevance_score", None),
    )


def _split_phrase_into_words(phrase: str) -> list[str]:
    """Split a search phrase into its lowercase words, in order."""
    return [word for word in _NON_ALPHANUMERIC.split(phrase.lower()) if word]


def _split_path_into_words(asset_path: str) -> frozenset[str]:
    """Split an asset path into lowercase words, also splitting camelCase."""
    # e.g. trashcan_wheeled_green_a01_01.usd -> trashcan wheeled green a01 01
    spaced = _CAMEL_CASE_BOUNDARY.sub(" ", asset_path)
    return frozenset(word for word in _NON_ALPHANUMERIC.split(spaced.lower()) if word)


def _path_words_match_singular_or_plural(word: str, path_words: frozenset[str]) -> bool:
    """True if the word is one of the path words. A trailing "s" on either side is ignored."""
    if word in path_words or f"{word}s" in path_words:
        return True
    return word.endswith("s") and word[:-1] in path_words


def _count_phrase_words_in_path(phrase: str, asset_path: str) -> int:
    """Count how many words of the phrase appear as whole words in the asset path."""
    path_words = _split_path_into_words(asset_path)
    return sum(1 for word in _split_phrase_into_words(phrase) if _path_words_match_singular_or_plural(word, path_words))
