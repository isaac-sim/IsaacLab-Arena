# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coordinate selected SimReady backends and merge their search results."""
from __future__ import annotations

import logging

from isaaclab_arena.agentic_environment_generation.simready_search_interface import (
    SimReadyCandidateCatalogue,
    SimReadySearchBackend,
    SimReadySearchConfig,
    SimReadySourceKind,
)

_logger = logging.getLogger(__name__)


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
            backend = _create_search_backend(source, config)
            result = backend.search(phrases, config.max_results_per_object)
        except Exception as exc:
            _logger.warning("simready backend %s failed: %s", source, exc)
            continue
        for candidate in result:
            key = (candidate.search_phrase, candidate.usd_path)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    matched = {candidate.search_phrase for candidate in candidates}
    return SimReadyCandidateCatalogue(
        candidates=candidates,
        unmatched_phrases=[phrase for phrase in phrases if phrase not in matched],
    )


def _create_search_backend(source: SimReadySourceKind, config: SimReadySearchConfig) -> SimReadySearchBackend:
    """Construct the implementation for one selected source."""
    if source == SimReadySourceKind.CENTRAL:
        from isaaclab_arena.agentic_environment_generation.simready_central_search import CentralSearchBackend

        return CentralSearchBackend()
    from isaaclab_arena.agentic_environment_generation.simready_asset_library_search import AssetLibrarySearchBackend

    return AssetLibrarySearchBackend(source, s3_url=config.s3_url, service_url=config.service_url)
