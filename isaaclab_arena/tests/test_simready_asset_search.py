# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SimReady asset search."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
    SimReadyCandidateCatalogue,
    SimReadyObjectCandidate,
    SimReadySearchConfig,
    SimReadySourceKind,
    _count_matching_words,
    _keep_whole_word_matches,
    _split_path_into_words,
    search_simready_objects,
    simready_search_config_from_cli,
)

CABINET_PATH = "SimReady/Residential/Kitchen/Cabinets/Cabinet_D01/sm_fixture_cabinet_d01_01.usd"
TRASH_CAN_PATH = "SimReady/Residential/Garage/sm_trashCan_wheeled_green_a01_01.usd"


class _FakeMatch:
    def __init__(self, asset_path: str, relevance_score: float | None = None):
        self.asset_path = asset_path
        self.relevance_score = relevance_score


class _FakeLibrary:
    def __init__(self, matches: list[_FakeMatch] | None = None):
        self._matches = matches or []

    def search(self, include_all=None, include_any=None):
        return list(self._matches)


async def _fake_configure_library(config, traces):
    return _FakeLibrary([_FakeMatch("s3://bucket/red_hammer.usd", relevance_score=0.95)])


def test_split_path_into_words_splits_camel_case_and_separators():
    words = _split_path_into_words(TRASH_CAN_PATH)
    assert {"trash", "can", "wheeled", "green", "garage"} <= words
    # "trashCan" must not survive as one token, or camelCase names stay unsearchable.
    assert "trashcan" not in words


def test_count_matching_words_ignores_substring_hits():
    # "bin" only appears inside "Cabinets", which is how a cabinet got picked for a grey bin.
    assert _count_matching_words("grey bin", CABINET_PATH) == 0
    assert _count_matching_words("kitchen cabinet", CABINET_PATH) == 2


def test_count_matching_words_tolerates_plurals():
    assert _count_matching_words("cabinet", CABINET_PATH) == 1
    assert _count_matching_words("cabinets", CABINET_PATH) == 1


def test_keep_whole_word_matches_filters_and_ranks():
    cabinet = _FakeMatch(CABINET_PATH)
    trash_can = _FakeMatch(TRASH_CAN_PATH)
    assert _keep_whole_word_matches([cabinet], "grey bin") == []
    # "green trash can" matches three words in the trash can and none in the cabinet.
    assert _keep_whole_word_matches([cabinet, trash_can], "green trash can") == [trash_can]


def test_keep_whole_word_matches_orders_by_word_overlap():
    cabinet = _FakeMatch(CABINET_PATH)
    trash_can = _FakeMatch(TRASH_CAN_PATH)
    ranked = _keep_whole_word_matches([cabinet, trash_can], "kitchen green trash can")
    assert ranked == [trash_can, cabinet]


def test_simready_candidate_catalogue_to_catalog_string():
    candidate = SimReadyObjectCandidate(
        search_phrase="red hammer",
        usd_path="s3://bucket/red_hammer.usd",
        tags=("sim-ready", "red", "hammer"),
        relevance_score=0.9,
    )
    block = SimReadyCandidateCatalogue(candidates=[candidate]).to_catalog_string()
    assert "SIMREADY_OBJECT_CANDIDATES" in block
    assert SIMREADY_USD_OBJECT_REGISTRY_NAME in block
    assert "red hammer" in block
    assert "s3://bucket/red_hammer.usd" in block


def test_simready_search_config_from_cli_defaults():
    config = simready_search_config_from_cli(
        enabled=True,
        source=SimReadySourceKind.ISAAC_SIM_GA.value,
        s3_url=None,
        service_url=None,
        project_config_path=None,
        indexed_path=None,
        max_results_per_object=2,
        use_service_fallback=False,
    )
    assert config.enabled is True
    assert config.source == SimReadySourceKind.ISAAC_SIM_GA
    assert config.max_results_per_object == 2


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._search_phrase_async",
    new_callable=AsyncMock,
)
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
    new_callable=AsyncMock,
)
def test_search_simready_objects_returns_candidates(mock_configure, mock_search_phrase):
    mock_configure.return_value = _FakeLibrary()
    mock_search_phrase.return_value = [
        SimReadyObjectCandidate(
            search_phrase="ceramic bowl",
            usd_path="s3://bucket/bowl.usd",
        )
    ]
    traces: list[str] = []
    catalog = search_simready_objects(
        ["ceramic bowl"],
        SimReadySearchConfig(enabled=True, source=SimReadySourceKind.ISAAC_SIM_GA),
        traces,
    )
    assert len(catalog.candidates) == 1
    assert catalog.candidates[0].usd_path == "s3://bucket/bowl.usd"
    assert catalog.candidates[0].registry_name == SIMREADY_USD_OBJECT_REGISTRY_NAME


def test_configure_asset_library_records_missing_package():
    import asyncio
    import sys

    from isaaclab_arena.agentic_environment_generation.simready_asset_search import _configure_asset_library

    async def _run() -> tuple[object, list[str]]:
        traces: list[str] = []
        blocked = {
            name: module for name, module in sys.modules.items() if name == "simready" or name.startswith("simready.")
        }
        for name in blocked:
            del sys.modules[name]
        try:
            with patch.dict(sys.modules, {"simready": None, "simready.search": None}):
                result = await _configure_asset_library(SimReadySearchConfig(enabled=True), traces)
        finally:
            sys.modules.update(blocked)
        return result, traces

    result, traces = asyncio.run(_run())
    assert result is None
    assert any("simready-search is not installed" in line for line in traces)


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
    new_callable=AsyncMock,
)
def test_search_simready_objects_returns_empty_when_library_unavailable(mock_configure):
    mock_configure.return_value = None
    traces: list[str] = []
    catalog = search_simready_objects(
        ["hammer"],
        SimReadySearchConfig(enabled=True),
        traces,
    )
    assert catalog.candidates == []


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._search_phrase_async",
    new_callable=AsyncMock,
)
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
    new_callable=AsyncMock,
)
def test_search_simready_objects_records_no_matches(mock_configure, mock_search_phrase):
    mock_configure.return_value = _FakeLibrary()
    mock_search_phrase.return_value = []
    traces: list[str] = []
    catalog = search_simready_objects(
        ["missing object"],
        SimReadySearchConfig(enabled=True),
        traces,
    )
    assert catalog.candidates == []
    assert any("no matches" in line for line in traces)
