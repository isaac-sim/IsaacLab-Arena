# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Match requested object names against SimReady asset names."""

import re

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def split_phrase_into_words(phrase: str) -> list[str]:
    """Split a search phrase into its lowercase words, in order."""
    return [word for word in _NON_ALPHANUMERIC.split(phrase.lower()) if word]


def split_path_into_words(asset_path: str) -> frozenset[str]:
    """Split an asset path into lowercase words, also splitting camelCase."""
    # e.g. trashcan_wheeled_green_a01_01.usd -> trashcan wheeled green a01 01
    spaced = _CAMEL_CASE_BOUNDARY.sub(" ", asset_path)
    return frozenset(word for word in _NON_ALPHANUMERIC.split(spaced.lower()) if word)


def path_words_match_singular_or_plural(word: str, path_words: frozenset[str]) -> bool:
    """True if the word is one of the path words. A trailing "s" on either side is ignored."""
    if word in path_words or f"{word}s" in path_words:
        return True
    return word.endswith("s") and word[:-1] in path_words


def count_phrase_words_in_path(phrase: str, asset_path: str) -> int:
    """Count how many words of the phrase appear as whole words in the asset path."""
    path_words = split_path_into_words(asset_path)
    return sum(1 for word in split_phrase_into_words(phrase) if path_words_match_singular_or_plural(word, path_words))


def asset_name_matches_object(phrase: str, asset_name: str) -> bool:
    """Require the final object word, allowing plural and joined compound names.

    This is a conservative lexical check, not semantic classification. For example,
    "grey bin" cannot match "grey cabinet", while "spray gun" can match "spraygun".
    """
    words = split_phrase_into_words(phrase)
    if not words:
        return False
    asset_words = split_path_into_words(asset_name)
    if path_words_match_singular_or_plural(words[-1], asset_words):
        return True
    return len(words) >= 2 and path_words_match_singular_or_plural("".join(words[-2:]), asset_words)
