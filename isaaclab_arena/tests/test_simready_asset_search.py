# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SimReady asset search."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from isaaclab_arena.agentic_environment_generation.simready_asset_search import search_simready_objects
from isaaclab_arena.agentic_environment_generation.simready_ga_assets import (
    _count_phrase_words_in_path,
    _filter_and_rank_path_matches,
    _has_single_rigid_body,
    _split_path_into_words,
)
from isaaclab_arena.agentic_environment_generation.simready_search_interface import (
    SimReadyObjectCandidate,
    SimReadySearchConfig,
    SimReadySourceKind,
    simready_search_config_from_cli,
)
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME

SEARCH_LOGGER = "isaaclab_arena.agentic_environment_generation.simready_ga_assets"
"""Where the search reports its progress, now that the agent's traces carry only errors."""

CABINET_PATH = "SimReady/Residential/Kitchen/Cabinets/Cabinet_D01/sm_fixture_cabinet_d01_01.usd"
GREY_CABINET_PATH = "SimReady/Residential/Kitchen/Cabinets/Cabinet_D01/sm_fixture_cabinet_grey_d01_01.usd"
TRASH_CAN_PATH = "SimReady/Residential/Garage/sm_trashCan_wheeled_green_a01_01.usd"
PLAIN_TRASH_CAN_PATH = "SimReady/Residential/Kitchen/sm_trashCan_a01_01.usd"

RIGID_BODY_CHECK_TARGET = "isaaclab_arena.agentic_environment_generation.simready_ga_assets._has_single_rigid_body"
# Patched where it is looked up, not where it is defined: simready_ga_assets binds the name at
# import time, so patching the defining module would leave that binding untouched.
RIGID_BODY_PATHS_TARGET = "isaaclab_arena.agentic_environment_generation.simready_ga_assets.read_asset_rigid_body_paths"
PATH_FILTERS_TARGET = "isaaclab_arena.agentic_environment_generation.simready_ga_assets._build_path_filters_from_phrase"
# Stubbed wherever the fake library stands in for a real one: building a path filter needs the
# simready package, which only ships with Isaac Sim, and the fake library ignores the filters.


class _FakeMatch:
    def __init__(self, asset_path: str, relevance_score: float | None = None):
        self.asset_path = asset_path
        self.relevance_score = relevance_score


class _FakeLibrary:
    def __init__(self, matches: list[_FakeMatch] | None = None):
        self._matches = matches or []

    def search(self, include_all=None, include_any=None):
        return list(self._matches)


def test_split_path_into_words_splits_camel_case_and_separators():
    words = _split_path_into_words(TRASH_CAN_PATH)
    assert {"trash", "can", "wheeled", "green", "garage"} <= words
    # "trashCan" must not survive as one token, or camelCase names stay unsearchable.
    assert "trashcan" not in words


def test_count_phrase_words_in_path_ignores_substring_hits():
    # "bin" only appears inside "Cabinets", which is how a cabinet got picked for a grey bin.
    assert _count_phrase_words_in_path("grey bin", CABINET_PATH) == 0
    assert _count_phrase_words_in_path("kitchen cabinet", CABINET_PATH) == 2


def test_count_phrase_words_in_path_tolerates_plurals():
    assert _count_phrase_words_in_path("cabinet", CABINET_PATH) == 1
    assert _count_phrase_words_in_path("cabinets", CABINET_PATH) == 1


def test_filter_and_rank_path_matches_filters_and_ranks():
    cabinet = _FakeMatch(CABINET_PATH)
    trash_can = _FakeMatch(TRASH_CAN_PATH)
    assert _filter_and_rank_path_matches([cabinet], "grey bin") == []
    # "green trash can" matches three words in the trash can and none in the cabinet.
    assert _filter_and_rank_path_matches([cabinet, trash_can], "green trash can") == [trash_can]


def test_filter_and_rank_path_matches_orders_by_word_overlap():
    green_can = _FakeMatch(TRASH_CAN_PATH)
    plain_can = _FakeMatch(PLAIN_TRASH_CAN_PATH)
    ranked = _filter_and_rank_path_matches([plain_can, green_can], "green trash can")
    assert ranked == [green_can, plain_can]


def test_filter_and_rank_path_matches_rejects_an_asset_that_only_shares_a_describing_word():
    grey_cabinet = _FakeMatch(GREY_CABINET_PATH)
    # A grey cabinet is not a grey bin: the colour matches but the object itself does not, and
    # handing back the cabinet is how a bin became a cabinet in a generated environment.
    assert _filter_and_rank_path_matches([grey_cabinet], "grey bin") == []
    assert _filter_and_rank_path_matches([grey_cabinet], "grey cabinet") == [grey_cabinet]


def test_has_single_rigid_body_accepts_a_single_rigid_body():
    with patch(RIGID_BODY_PATHS_TARGET, return_value=["/Asset/bottle"]):
        assert _has_single_rigid_body("s3://bucket/bottle.usd") == (True, "")


def test_has_single_rigid_body_turns_down_several_rigid_bodies():
    # A bottle and its cap are two bodies even though a fixed joint holds them together.
    with patch(RIGID_BODY_PATHS_TARGET, return_value=["/Asset/bottle", "/Asset/cap"]):
        assert _has_single_rigid_body("s3://bucket/bottle.usd") == (False, "it has 2 rigid bodies")


def test_has_single_rigid_body_turns_down_an_asset_without_physics():
    with patch(RIGID_BODY_PATHS_TARGET, return_value=[]):
        assert _has_single_rigid_body("s3://bucket/decoration.usd") == (False, "it has no rigid body")


def test_has_single_rigid_body_turns_down_an_unreadable_asset():
    with patch(RIGID_BODY_PATHS_TARGET, side_effect=FileNotFoundError("no such file")):
        is_valid, reason = _has_single_rigid_body("s3://bucket/missing.usd")
    assert not is_valid
    assert "could not be read" in reason


@patch(PATH_FILTERS_TARGET, return_value=[])
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._create_asset_library_for_source",
)
def test_search_falls_back_to_the_next_hit_when_one_is_rejected(mock_configure, mock_path_filters, caplog):
    mock_configure.return_value = _FakeLibrary([_FakeMatch(TRASH_CAN_PATH), _FakeMatch(PLAIN_TRASH_CAN_PATH)])
    # The green one is named after more of the phrase, so it is looked at first and turned down.
    verdicts = {TRASH_CAN_PATH: (False, "it has 2 rigid bodies"), PLAIN_TRASH_CAN_PATH: (True, "")}
    with patch(RIGID_BODY_CHECK_TARGET, side_effect=lambda usd_path: verdicts[usd_path]):
        with caplog.at_level(logging.INFO, logger=SEARCH_LOGGER):
            catalog = search_simready_objects(["green trash can"], SimReadySearchConfig())
    assert [candidate.usd_path for candidate in catalog.candidates] == [PLAIN_TRASH_CAN_PATH]
    assert catalog.unmatched_phrases == []
    assert TRASH_CAN_PATH in caplog.text and "2 rigid bodies" in caplog.text


@patch(PATH_FILTERS_TARGET, return_value=[])
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._create_asset_library_for_source",
)
def test_search_reports_a_phrase_with_only_rejected_hits_as_unmatched(mock_configure, mock_path_filters, caplog):
    mock_configure.return_value = _FakeLibrary([_FakeMatch(CABINET_PATH)])
    with patch(RIGID_BODY_CHECK_TARGET, return_value=(False, "it has no rigid body")):
        with caplog.at_level(logging.INFO, logger=SEARCH_LOGGER):
            catalog = search_simready_objects(["kitchen cabinet"], SimReadySearchConfig())
    assert catalog.candidates == []
    assert catalog.unmatched_phrases == ["kitchen cabinet"]
    assert "no usable asset" in caplog.text


def test_simready_search_config_from_cli_defaults():
    config = simready_search_config_from_cli(
        sources=[SimReadySourceKind.ISAAC_SIM_GA.value],
        s3_url=None,
        service_url=None,
        max_results_per_object=2,
    )
    assert config.sources == [SimReadySourceKind.ISAAC_SIM_GA]
    assert config.max_results_per_object == 2


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._search_library_rigid_body_candidates",
)
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._create_asset_library_for_source",
)
def test_search_simready_objects_returns_candidates(mock_configure, mock_search_library_rigid_body_candidates):
    mock_configure.return_value = _FakeLibrary()
    mock_search_library_rigid_body_candidates.return_value = [
        SimReadyObjectCandidate(
            search_phrase="ceramic bowl",
            usd_path="s3://bucket/bowl.usd",
        )
    ]
    catalog = search_simready_objects(["ceramic bowl"], SimReadySearchConfig(sources=[SimReadySourceKind.ISAAC_SIM_GA]))
    assert len(catalog.candidates) == 1
    assert catalog.candidates[0].usd_path == "s3://bucket/bowl.usd"
    assert catalog.candidates[0].registry_name == SIMREADY_USD_OBJECT_REGISTRY_NAME


def test_create_asset_library_for_source_records_an_unknown_source(caplog):
    from isaaclab_arena.agentic_environment_generation.simready_ga_assets import _create_asset_library_for_source

    config = SimReadySearchConfig()
    with caplog.at_level(logging.ERROR, logger=SEARCH_LOGGER):
        assert (
            _create_asset_library_for_source("not-a-source", s3_url=config.s3_url, service_url=config.service_url)
            is None
        )
    assert "unknown simready source" in caplog.text


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._create_asset_library_for_source",
)
def test_search_simready_objects_returns_empty_when_library_unavailable(mock_configure):
    mock_configure.return_value = None
    catalog = search_simready_objects(["hammer"], SimReadySearchConfig())
    assert catalog.candidates == []


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._search_library_rigid_body_candidates",
)
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_ga_assets._create_asset_library_for_source",
)
def test_search_simready_objects_records_no_matches(mock_configure, mock_search_library_rigid_body_candidates, caplog):
    mock_configure.return_value = _FakeLibrary()
    mock_search_library_rigid_body_candidates.return_value = []
    with caplog.at_level(logging.INFO, logger=SEARCH_LOGGER):
        catalog = search_simready_objects(["missing object"], SimReadySearchConfig())
    assert catalog.candidates == []
    assert catalog.unmatched_phrases == ["missing object"]
    assert "no usable asset" in caplog.text


def test_multiple_sources_merge_and_deduplicate_candidates():
    shared = SimReadyObjectCandidate(search_phrase="egg", usd_path="shared.usd")
    central = SimReadyObjectCandidate(search_phrase="egg", usd_path="central.usd")
    with (
        patch(f"{SEARCH_LOGGER}._create_asset_library_for_source", return_value=object()),
        patch(f"{SEARCH_LOGGER}._search_library_rigid_body_candidates", side_effect=[[shared], []]),
        patch(
            "isaaclab_arena.agentic_environment_generation.simready_central_assets.search_central_objects",
            side_effect=[[shared, central], []],
        ),
    ):
        result = search_simready_objects(
            ["egg", "missing"],
            SimReadySearchConfig(sources=[SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.CENTRAL]),
        )
    assert result.candidates == [shared, central]
    assert result.unmatched_phrases == ["missing"]


def test_multiple_sources_keep_central_when_library_initialization_fails():
    central = SimReadyObjectCandidate(search_phrase="egg", usd_path="central.usd")
    with (
        patch(f"{SEARCH_LOGGER}._create_asset_library_for_source", side_effect=RuntimeError("unavailable")),
        patch(
            "isaaclab_arena.agentic_environment_generation.simready_central_assets.search_central_objects",
            return_value=[central],
        ),
    ):
        result = search_simready_objects(
            ["egg"], SimReadySearchConfig(sources=[SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.CENTRAL])
        )
    assert result.candidates == [central]
    assert not result.unmatched_phrases


def test_multiple_sources_keep_library_when_central_fails():
    candidate = SimReadyObjectCandidate(search_phrase="egg", usd_path="library.usd")
    with (
        patch(f"{SEARCH_LOGGER}._create_asset_library_for_source", return_value=object()),
        patch(f"{SEARCH_LOGGER}._search_library_rigid_body_candidates", return_value=[candidate]),
        patch(
            "isaaclab_arena.agentic_environment_generation.simready_central_assets.search_central_objects",
            side_effect=RuntimeError("unavailable"),
        ),
    ):
        result = search_simready_objects(
            ["egg"], SimReadySearchConfig(sources=[SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.CENTRAL])
        )
    assert result.candidates == [candidate]
    assert not result.unmatched_phrases


def test_source_list_searches_selected_backends_once():
    config = simready_search_config_from_cli(
        sources=["service", "simready-central", "service"],
        s3_url=None,
        service_url="http://asset-service",
        max_results_per_object=2,
    )
    assert config.sources == [SimReadySourceKind.SERVICE, SimReadySourceKind.CENTRAL]
    with (
        patch(f"{SEARCH_LOGGER}._create_asset_library_for_source", return_value=object()) as configure,
        patch(f"{SEARCH_LOGGER}._search_library_rigid_body_candidates", return_value=[]) as library_search,
        patch(
            "isaaclab_arena.agentic_environment_generation.simready_central_assets.search_central_objects",
            return_value=[],
        ) as central_search,
    ):
        result = search_simready_objects(["egg"], config)
    configure.assert_called_once_with(SimReadySourceKind.SERVICE, s3_url=config.s3_url, service_url=config.service_url)
    library_search.assert_called_once()
    central_search.assert_called_once_with("egg", max_results=2)
    assert result.unmatched_phrases == ["egg"]


@pytest.mark.parametrize("source", [SimReadySourceKind.ISAAC_SIM_GA, SimReadySourceKind.CENTRAL])
def test_backend_keeps_other_phrase_results_when_one_phrase_fails(source):
    candidate = SimReadyObjectCandidate(search_phrase="wrench", usd_path="wrench.usd")
    target = (
        f"{SEARCH_LOGGER}._search_library_rigid_body_candidates"
        if source == SimReadySourceKind.ISAAC_SIM_GA
        else "isaaclab_arena.agentic_environment_generation.simready_central_assets.search_central_objects"
    )
    with (
        patch(f"{SEARCH_LOGGER}._create_asset_library_for_source", return_value=object()),
        patch(target, side_effect=[RuntimeError("unavailable"), [candidate]]) as search,
    ):
        result = search_simready_objects(["missing", "wrench"], SimReadySearchConfig(sources=[source]))
    assert search.call_count == 2
    assert result.candidates == [candidate]
    assert result.unmatched_phrases == ["missing"]
