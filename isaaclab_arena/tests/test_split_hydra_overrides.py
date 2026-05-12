# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`isaaclab_arena_environments.cli.split_hydra_overrides`.

Plain-Python unit tests: do not require Isaac Sim. Cover both halves of the
contract documented on the helper:

* Hydra-shaped tokens (``key.path=value``, ``+key=value``, ``++key=value``,
  ``~key``) are passed through unchanged and in order.
* Anything else left over from ``parse_known_args`` (a stray ``--flag``, an
  unbound positional, a typo'd flag) routes through ``parser.error`` and
  exits the process with code 2 -- matching the strict ``parse_args``
  behaviour the entry-point scripts had before the switch.
"""

import argparse

import pytest

from isaaclab_arena_environments.cli import split_hydra_overrides


def _parser() -> argparse.ArgumentParser:
    """Minimal parser standing in for any of the entry-point parsers."""
    return argparse.ArgumentParser(prog="test_runner")


def test_accepts_all_four_hydra_shapes():
    """All four shapes named in the helper docstring pass through unchanged."""
    tokens = [
        "cracker_box.color.enabled=true",
        "+cracker_box.color.sampler.low=[0.2,0.2,0.0]",
        "++tomato_soup_can.color.sampler.high=[0.0,1.0,1.0]",
        "~lighting.intensity",
        "~lighting.intensity=0.5",  # `~` delete with explicit value is also legal
    ]
    assert split_hydra_overrides(tokens, _parser()) == tokens


def test_empty_rhs_accepted():
    """Empty value (`a.b=`) is a valid Hydra null / empty-string assignment."""
    assert split_hydra_overrides(["a.b="], _parser()) == ["a.b="]


def test_preserves_order():
    """Order is preserved so Hydra's later-wins semantics still apply."""
    tokens = [
        "a.b=1",
        "a.b=2",
        "a.c=3",
    ]
    assert split_hydra_overrides(tokens, _parser()) == tokens


def test_empty_input_returns_empty_list():
    """``parse_known_args`` returning ``[]`` is the common case -- no error."""
    assert split_hydra_overrides([], _parser()) == []


@pytest.mark.parametrize(
    "bad_token",
    [
        "--object",  # typo'd flag (the real one is --object on a subparser, but here it's unknown)
        "--unknown_flag",  # bare unknown flag
        "stray_positional",  # bare positional (no '=' so not a Hydra set)
        "1.0",  # numeric -- not a valid key
        "key with space=value",  # whitespace not allowed in key
        "=value_only",  # missing key
        "+just_plus",  # `+` prefix without `=value` is not a delete (`~` is)
        "",  # empty token
    ],
)
def test_rejects_non_hydra_token(bad_token):
    """A non-Hydra leftover must cause the parser to exit with non-zero status."""
    parser = _parser()
    with pytest.raises(SystemExit) as exc_info:
        split_hydra_overrides([bad_token], parser)
    # argparse.error uses exit code 2.
    assert exc_info.value.code == 2


def test_rejects_when_mixed_with_valid_tokens():
    """A single bad token poisons the whole batch -- we don't silently drop it."""
    parser = _parser()
    with pytest.raises(SystemExit) as exc_info:
        split_hydra_overrides(["cracker_box.color.enabled=true", "--object"], parser)
    assert exc_info.value.code == 2


def test_error_message_names_bad_tokens(capsys):
    """The error message includes the offending token(s) so users can fix typos."""
    parser = _parser()
    with pytest.raises(SystemExit):
        split_hydra_overrides(["--typo", "stray"], parser)
    err = capsys.readouterr().err
    assert "--typo" in err
    assert "stray" in err
