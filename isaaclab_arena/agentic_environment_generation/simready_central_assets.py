# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Search SimReady Central through its specialized REST API."""
from __future__ import annotations

import logging

import requests

from isaaclab_arena.agentic_environment_generation.simready_search_interface import (
    MIN_INSPECTED_MATCHES_PER_PHRASE,
    SimReadyObjectCandidate,
)
from isaaclab_arena.assets.simready_central_asset_loader import download_and_wrap_central_asset

CENTRAL_SEARCH_URL = "https://nvidia-simready-central.hf.space/search_hybrid"
_logger = logging.getLogger(__name__)


class CentralSearchBackend:
    """Search Central and validate downloaded single-rigid-body packages."""

    def search(self, phrases: list[str], max_results: int) -> list[SimReadyObjectCandidate]:
        """Search each phrase independently so one request failure does not discard other hits."""
        candidates = []
        for phrase in phrases:
            try:
                hits = search_central_objects(phrase, max_results=max_results)
                if not hits:
                    _logger.info("Central found no usable asset for %r", phrase)
                candidates.extend(hits)
            except Exception as exc:
                _logger.warning("Central search failed for %r: %s", phrase, exc)
        return candidates


def search_central_objects(phrase: str, *, max_results: int) -> list[SimReadyObjectCandidate]:
    """Search Central and retain only locally validated single-body props."""
    assert max_results > 0, "max_results must be positive"
    response = requests.post(
        CENTRAL_SEARCH_URL,
        headers={"x-usdsearch-storage-backend": "usdsearch-dev"},
        json={
            "hybrid_text_query": phrase,
            "file_extension_include": "usd*",
            "filter_by_properties_numeric": "simready_metadata_rigid_body_count=1",
            "return_usd_properties": True,
            "limit": max(MIN_INSPECTED_MATCHES_PER_PHRASE, max_results),
        },
        timeout=30,
    )
    response.raise_for_status()
    candidates = []
    for hit in response.json()["hits"]:
        try:
            source = hit["source"]
            # Fail closed even if a backend ignores the numeric filter.
            counts = [
                prop.get("value_numeric", prop.get("value"))
                for prop in source.get("usd_properties", [])
                if prop.get("name") == "simready_metadata_rigid_body_count"
            ]
            if not counts or any(float(count) != 1 for count in counts):
                continue
            usd_path = download_and_wrap_central_asset(source["path"])
            candidates.append(
                SimReadyObjectCandidate(
                    search_phrase=phrase,
                    usd_path=usd_path,
                    tags=("sim-ready", "single-rigid-body"),
                    relevance_score=hit.get("rrf_score"),
                    source_url=source["path"].replace("hf://", "https://", 1),
                )
            )
        except Exception as exc:
            _logger.info("Skipping Central asset %s: %s", hit.get("id"), exc)
            continue
        if len(candidates) >= max_results:
            break
    return candidates
