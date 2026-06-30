# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The opt-in Maple staging asset override must not leak into the registered (shared) asset class."""

from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    _PROD_NUCLEUS_HOST,
    _STAGING_NUCLEUS_HOST,
    _staging_subclass,
    _to_staging_url,
)


def test_pick_targets_cli_is_fail_closed():
    """--pick_targets uses nargs='+' / default None: absent -> single-object; a present flag must carry names."""
    import argparse

    import pytest

    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironment

    parser = argparse.ArgumentParser()
    PickAndPlaceMapleTableEnvironment.add_cli_args(parser)

    # Absent -> None (stock single-object path, unchanged).
    assert parser.parse_args([]).pick_targets is None
    # Present with names -> ordered list.
    assert parser.parse_args(["--pick_targets", "a", "b", "c"]).pick_targets == ["a", "b", "c"]
    # Present but empty is rejected (nargs='+'), so it cannot silently fall back to single-object.
    with pytest.raises(SystemExit):
        parser.parse_args(["--pick_targets"])


def test_droid_stand_staging_override_is_instance_local():
    """Overriding one DROID embodiment's stand to staging must NOT leak into a fresh (stock) embodiment.

    The stand AssetBaseCfg is a class-level configclass default; the override must deep-copy it, not mutate
    the shared object in place. (Deferred imports: instantiating the embodiment needs the booted app.)
    """
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import _apply_staging_stand_override

    staged_emb = DroidAbsoluteJointPositionEmbodiment(enable_cameras=False)
    prod_url = staged_emb.scene_config.stand.spawn.usd_path
    assert _PROD_NUCLEUS_HOST in prod_url

    staged_url = _apply_staging_stand_override(staged_emb)
    assert _STAGING_NUCLEUS_HOST in staged_url
    assert staged_emb.scene_config.stand.spawn.usd_path == staged_url

    # A fresh stock embodiment built afterward must still point at production — no leak via the shared default.
    stock_emb = DroidAbsoluteJointPositionEmbodiment(enable_cameras=False)
    assert stock_emb.scene_config.stand.spawn.usd_path == prod_url, "staging override leaked into a stock embodiment"
    assert _PROD_NUCLEUS_HOST in stock_emb.scene_config.stand.spawn.usd_path


def test_to_staging_url_swaps_host_only():
    prod = f"https://{_PROD_NUCLEUS_HOST}/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/x/maple_table.usda"
    staged = _to_staging_url(prod)
    assert _STAGING_NUCLEUS_HOST in staged and _PROD_NUCLEUS_HOST not in staged
    # Only the host changes; the asset path is preserved exactly.
    assert staged.split(_STAGING_NUCLEUS_HOST, 1)[1] == prod.split(_PROD_NUCLEUS_HOST, 1)[1]


def test_staging_subclass_does_not_mutate_registered_class():
    """_staging_subclass returns a subclass with the staging URL; the input class is left untouched."""

    class _FakeAsset:
        usd_path = f"https://{_PROD_NUCLEUS_HOST}/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/x/maple_table.usda"

    original = _FakeAsset.usd_path
    staged_cls, staged_url = _staging_subclass(_FakeAsset)

    # The subclass points at staging...
    assert staged_cls is not _FakeAsset
    assert issubclass(staged_cls, _FakeAsset)
    assert staged_cls.usd_path == staged_url
    assert _STAGING_NUCLEUS_HOST in staged_url
    # ...but the registered (shared) class is unchanged — no leak into later non-staging jobs/rebuilds.
    assert _FakeAsset.usd_path == original
    assert _PROD_NUCLEUS_HOST in _FakeAsset.usd_path
