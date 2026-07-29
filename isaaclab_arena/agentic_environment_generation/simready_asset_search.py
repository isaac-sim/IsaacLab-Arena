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

SIMREADY_USD_OBJECT_REGISTRY_NAME = "simready_usd_object"

ISAAC_SIMREADY_GA_S3_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady"
)
DEFAULT_SIMREADY_SERVICE_URL = "https://search.simready.omniverse.nvidia.com/"


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
    use_service_fallback: bool = False


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

    def params(self) -> dict[str, Any]:
        return {"usd_path": self.usd_path, "tags": list(self.tags)}


@dataclass
class SimReadyCandidateCatalogue:
    """Prompt-scoped SimReady hits for spec inference."""

    candidates: list[SimReadyObjectCandidate] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format candidates as the SIMREADY_OBJECT_CANDIDATES prompt block."""
        if not self.candidates:
            return ""
        lines = [
            f"SIMREADY_OBJECT_CANDIDATES ({len(self.candidates)}):",
            (
                "When a desired object matches a candidate below, use "
                f"registry_name={SIMREADY_USD_OBJECT_REGISTRY_NAME!r} and copy params exactly."
            ),
        ]
        for index, candidate in enumerate(self.candidates, start=1):
            tag_text = ", ".join(candidate.tags)
            score = f" relevance={candidate.relevance_score:.2f}" if candidate.relevance_score is not None else ""
            lines.append(
                f"- candidate_{index}: requested_object={candidate.search_phrase!r}{score}\n"
                f"  registry_name: {candidate.registry_name}\n"
                "  params:\n"
                f"    usd_path: {candidate.usd_path}\n"
                f"    tags: [{tag_text}]"
            )
        return "\n".join(lines)


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _slugify_phrase(phrase: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower()).strip("_")
    return slug or "object"


def _phrase_words(phrase: str) -> list[str]:
    return [word for word in re.split(r"\s+", phrase.strip().lower()) if word]


def _phrase_path_filters(phrase: str) -> list[Any]:
    from simready.search import SearchFilterPathContains

    words = _phrase_words(phrase)
    if not words:
        return [SearchFilterPathContains(phrase)]
    return [SearchFilterPathContains(word) for word in words]


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
    """Keep only results that share a whole word with the phrase, best matches first.

    ``SearchFilterPathContains`` matches any substring, so searching for "bin" returns every
    cabinet in the library. Asking for whole words drops those results.
    """
    scored = [(_count_matching_words(phrase, str(match.asset_path)), match) for match in matches]
    kept = [entry for entry in scored if entry[0] > 0]
    # The sort is stable, so results with the same number of matching words keep their order.
    kept.sort(key=lambda entry: entry[0], reverse=True)
    return [match for _, match in kept]


def _select_matches(matches: list[Any], max_results: int) -> list[Any]:
    if not matches:
        return []
    if matches and getattr(matches[0], "relevance_score", None) is not None:
        matches = sorted(
            matches,
            key=lambda match: match.relevance_score if match.relevance_score is not None else 0.0,
            reverse=True,
        )
    return matches[:max_results]


async def _configure_asset_library(config: SimReadySearchConfig, traces: list[str]) -> Any | None:
    try:
        from simready.search import AssetLibrary
    except ImportError:
        traces.append("simready-search is not installed; install with pip install 'isaaclab_arena[simready]'")
        return None

    library = AssetLibrary()
    source = config.source
    if source == SimReadySourceKind.ISAAC_SIM_GA:
        await library.add_s3_source(config.s3_url or ISAAC_SIMREADY_GA_S3_URL)
        return library
    if source == SimReadySourceKind.S3:
        assert config.s3_url, "simready s3 source requires s3_url"
        await library.add_s3_source(config.s3_url)
        return library
    if source == SimReadySourceKind.SERVICE:
        library.add_service_source(config.service_url or DEFAULT_SIMREADY_SERVICE_URL)
        return library
    if source == SimReadySourceKind.CACHE:
        assert config.project_config_path, "simready cache source requires project_config_path"
        await library.add_cache_source(config.project_config_path)
        return library
    if source == SimReadySourceKind.INDEXED:
        assert config.indexed_path, "simready indexed source requires indexed_path"
        await library.add_indexed_source(config.indexed_path, config.indexed_directory_type)
        return library
    traces.append(f"unknown simready source: {source}")
    return None


async def _search_phrase_async(
    library: Any,
    phrase: str,
    *,
    use_service_phrase: bool,
    service_url: str,
    max_results: int,
) -> list[SimReadyObjectCandidate]:
    from simready.search import AssetLibrary, SearchFilterPhrase

    matches = _keep_whole_word_matches(library.search(include_any=_phrase_path_filters(phrase)), phrase)
    if not matches and use_service_phrase:
        service_library = AssetLibrary()
        service_library.add_service_source(service_url)
        matches = service_library.search(include_all=[SearchFilterPhrase(phrase)])

    candidates: list[SimReadyObjectCandidate] = []
    for match in _select_matches(matches, max_results):
        usd_path = str(match.asset_path)
        tags = ("sim-ready", *_slugify_phrase(phrase).split("_"))
        tags = tuple(dict.fromkeys(tag for tag in tags if tag))
        candidates.append(
            SimReadyObjectCandidate(
                search_phrase=phrase,
                usd_path=usd_path,
                tags=tags,
                relevance_score=getattr(match, "relevance_score", None),
            )
        )
    return candidates


async def search_simready_objects_async(
    object_phrases: list[str],
    config: SimReadySearchConfig,
    traces: list[str],
) -> SimReadyCandidateCatalogue:
    """Query SimReady for each normalized object phrase."""
    phrases = [phrase.strip() for phrase in object_phrases if phrase.strip()]
    if not phrases:
        return SimReadyCandidateCatalogue()

    library = await _configure_asset_library(config, traces)
    if library is None:
        return SimReadyCandidateCatalogue()

    candidates: list[SimReadyObjectCandidate] = []
    for phrase in phrases:
        try:
            hits = await _search_phrase_async(
                library,
                phrase,
                use_service_phrase=config.use_service_fallback or config.source == SimReadySourceKind.SERVICE,
                service_url=config.service_url,
                max_results=config.max_results_per_object,
            )
        except Exception as exc:
            traces.append(f"simready search failed for {phrase!r}: {exc}")
            continue
        if not hits:
            traces.append(f"simready search returned no matches for {phrase!r}")
            continue
        candidates.extend(hits)

    return SimReadyCandidateCatalogue(candidates=candidates)


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
    use_service_fallback: bool,
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
        use_service_fallback=use_service_fallback,
    )
