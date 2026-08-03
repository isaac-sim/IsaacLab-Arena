# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SimReady asset search."""

from __future__ import annotations

import logging
from unittest.mock import patch

from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadyObjectCandidate,
    SimReadySearchConfig,
    SimReadySourceKind,
    _count_matching_words,
    _is_valid_isaaclab_rigidbody,
    _keep_whole_word_matches,
    _split_path_into_words,
    search_simready_objects,
    simready_search_config_from_cli,
)
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME

SEARCH_LOGGER = "isaaclab_arena.agentic_environment_generation.simready_asset_search"
"""Where the search reports its progress, now that the agent's traces carry only errors."""

CABINET_PATH = "SimReady/Residential/Kitchen/Cabinets/Cabinet_D01/sm_fixture_cabinet_d01_01.usd"
GREY_CABINET_PATH = "SimReady/Residential/Kitchen/Cabinets/Cabinet_D01/sm_fixture_cabinet_grey_d01_01.usd"
TRASH_CAN_PATH = "SimReady/Residential/Garage/sm_trashCan_wheeled_green_a01_01.usd"
PLAIN_TRASH_CAN_PATH = "SimReady/Residential/Kitchen/sm_trashCan_a01_01.usd"

RIGID_BODY_CHECK_TARGET = (
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._is_valid_isaaclab_rigidbody"
)
# Patched where it is looked up, not where it is defined: simready_asset_search binds the name at
# import time, so patching the defining module would leave that binding untouched.
RIGID_BODY_PATHS_TARGET = (
    "isaaclab_arena.agentic_environment_generation.simready_asset_search.read_asset_rigid_body_paths"
)
PATH_FILTERS_TARGET = "isaaclab_arena.agentic_environment_generation.simready_asset_search._phrase_path_filters"
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
    green_can = _FakeMatch(TRASH_CAN_PATH)
    plain_can = _FakeMatch(PLAIN_TRASH_CAN_PATH)
    ranked = _keep_whole_word_matches([plain_can, green_can], "green trash can")
    assert ranked == [green_can, plain_can]


def test_keep_whole_word_matches_rejects_an_asset_that_only_shares_a_describing_word():
    grey_cabinet = _FakeMatch(GREY_CABINET_PATH)
    # A grey cabinet is not a grey bin: the colour matches but the object itself does not, and
    # handing back the cabinet is how a bin became a cabinet in a generated environment.
    assert _keep_whole_word_matches([grey_cabinet], "grey bin") == []
    assert _keep_whole_word_matches([grey_cabinet], "grey cabinet") == [grey_cabinet]


def test_is_valid_isaaclab_rigidbody_accepts_a_single_rigid_body():
    with patch(RIGID_BODY_PATHS_TARGET, return_value=["/Asset/bottle"]):
        assert _is_valid_isaaclab_rigidbody("s3://bucket/bottle.usd") == (True, "")


def test_is_valid_isaaclab_rigidbody_turns_down_several_rigid_bodies():
    # A bottle and its cap are two bodies even though a fixed joint holds them together.
    with patch(RIGID_BODY_PATHS_TARGET, return_value=["/Asset/bottle", "/Asset/cap"]):
        assert _is_valid_isaaclab_rigidbody("s3://bucket/bottle.usd") == (False, "it has 2 rigid bodies")


def test_is_valid_isaaclab_rigidbody_turns_down_an_asset_without_physics():
    with patch(RIGID_BODY_PATHS_TARGET, return_value=[]):
        assert _is_valid_isaaclab_rigidbody("s3://bucket/decoration.usd") == (False, "it has no rigid body")


def test_is_valid_isaaclab_rigidbody_turns_down_an_unreadable_asset():
    with patch(RIGID_BODY_PATHS_TARGET, side_effect=FileNotFoundError("no such file")):
        is_valid, reason = _is_valid_isaaclab_rigidbody("s3://bucket/missing.usd")
    assert not is_valid
    assert "could not be read" in reason


@patch(PATH_FILTERS_TARGET, return_value=[])
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
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
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
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
        source=SimReadySourceKind.ISAAC_SIM_GA.value,
        s3_url=None,
        service_url=None,
        max_results_per_object=2,
    )
    assert config.source == SimReadySourceKind.ISAAC_SIM_GA
    assert config.max_results_per_object == 2


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._search_phrase",
)
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
)
def test_search_simready_objects_returns_candidates(mock_configure, mock_search_phrase):
    mock_configure.return_value = _FakeLibrary()
    mock_search_phrase.return_value = [
        SimReadyObjectCandidate(
            search_phrase="ceramic bowl",
            usd_path="s3://bucket/bowl.usd",
        )
    ]
    catalog = search_simready_objects(["ceramic bowl"], SimReadySearchConfig(source=SimReadySourceKind.ISAAC_SIM_GA))
    assert len(catalog.candidates) == 1
    assert catalog.candidates[0].usd_path == "s3://bucket/bowl.usd"
    assert catalog.candidates[0].registry_name == SIMREADY_USD_OBJECT_REGISTRY_NAME


def test_configure_asset_library_records_an_unknown_source(caplog):
    from isaaclab_arena.agentic_environment_generation.simready_asset_search import _configure_asset_library

    config = SimReadySearchConfig()
    config.source = "not-a-source"
    with caplog.at_level(logging.ERROR, logger=SEARCH_LOGGER):
        assert _configure_asset_library(config) is None
    assert "unknown simready source" in caplog.text


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
)
def test_search_simready_objects_returns_empty_when_library_unavailable(mock_configure):
    mock_configure.return_value = None
    catalog = search_simready_objects(["hammer"], SimReadySearchConfig())
    assert catalog.candidates == []


@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._search_phrase",
)
@patch(
    "isaaclab_arena.agentic_environment_generation.simready_asset_search._configure_asset_library",
)
def test_search_simready_objects_records_no_matches(mock_configure, mock_search_phrase, caplog):
    mock_configure.return_value = _FakeLibrary()
    mock_search_phrase.return_value = []
    with caplog.at_level(logging.INFO, logger=SEARCH_LOGGER):
        catalog = search_simready_objects(["missing object"], SimReadySearchConfig())
    assert catalog.candidates == []
    assert catalog.unmatched_phrases == ["missing object"]
    assert "no usable asset" in caplog.text
