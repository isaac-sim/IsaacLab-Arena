# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for the physics-settle failure cache and its layout signature.

The cache persists which solved layouts failed to settle so future runs skip them. Correctness rests on
two guards: a uid only names the same layout when the draw order (seed) and the object composition
(signature) both match. These tests pin down the persistence round-trip, both invalidation paths, and
the signature's determinism/sensitivity -- none of which need a SimulationApp.
"""

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.physics_settle_failure_cache import PhysicsSettleFailureCache
from isaaclab_arena.relations.pooled_object_placer import compute_layout_signature
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _layouts(count):
    """Distinct dummy layout objects. The cache only keys on id(layout), so any object works."""
    return [object() for _ in range(count)]


def _register_all(cache, layouts):
    """Register layouts in order; return the known-failure flag each register() reports."""
    return [cache.register(layout) for layout in layouts]


def _isolate_cache_home(tmp_path, monkeypatch):
    """Point the cache's ``~/.cache`` lookup at a temp dir so tests never touch the real home."""
    monkeypatch.setenv("HOME", str(tmp_path))


# ----------------------------------------------------------------------------------------------------
# Persistence round-trip
# ----------------------------------------------------------------------------------------------------


def test_failures_persist_across_instances(tmp_path, monkeypatch):
    """A second cache with matching key/seed/signature replays the previously recorded failures by uid."""
    _isolate_cache_home(tmp_path, monkeypatch)
    key, seed, sig = "scene_a", 7, "sig_a"

    writer = PhysicsSettleFailureCache(cache_key=key, placement_seed=seed, layout_signature=sig)
    written = _layouts(5)
    _register_all(writer, written)
    writer.record_failures([written[1], written[3]])

    reader = PhysicsSettleFailureCache(cache_key=key, placement_seed=seed, layout_signature=sig)
    flags = _register_all(reader, _layouts(5))
    # uids 1 and 3 were recorded as failures, so register() flags exactly those enqueue positions.
    assert flags == [False, True, False, True, False]


def test_record_failures_noop_when_nothing_new(tmp_path, monkeypatch):
    """Recording only already-known failures writes nothing new (idempotent persistence)."""
    _isolate_cache_home(tmp_path, monkeypatch)
    key, seed, sig = "scene_b", 1, "sig_b"

    first = PhysicsSettleFailureCache(cache_key=key, placement_seed=seed, layout_signature=sig)
    layouts = _layouts(3)
    _register_all(first, layouts)
    first.record_failures([layouts[2]])
    first.record_failures([layouts[2]])  # same uid again

    second = PhysicsSettleFailureCache(cache_key=key, placement_seed=seed, layout_signature=sig)
    assert _register_all(second, _layouts(3)) == [False, False, True]


# ----------------------------------------------------------------------------------------------------
# Invalidation
# ----------------------------------------------------------------------------------------------------


def test_seed_mismatch_discards_cache(tmp_path, monkeypatch):
    """A different placement_seed means a different draw order, so the cached uids are ignored."""
    _isolate_cache_home(tmp_path, monkeypatch)
    key, sig = "scene_c", "sig_c"

    writer = PhysicsSettleFailureCache(cache_key=key, placement_seed=7, layout_signature=sig)
    layouts = _layouts(4)
    _register_all(writer, layouts)
    writer.record_failures([layouts[0], layouts[2]])

    other_seed = PhysicsSettleFailureCache(cache_key=key, placement_seed=8, layout_signature=sig)
    assert _register_all(other_seed, _layouts(4)) == [False, False, False, False]


def test_signature_mismatch_discards_cache(tmp_path, monkeypatch):
    """A different layout signature means a different object composition, so cached uids are ignored."""
    _isolate_cache_home(tmp_path, monkeypatch)
    key, seed = "scene_d", 3

    writer = PhysicsSettleFailureCache(cache_key=key, placement_seed=seed, layout_signature="sig_old")
    layouts = _layouts(4)
    _register_all(writer, layouts)
    writer.record_failures([layouts[0], layouts[2]])

    swapped = PhysicsSettleFailureCache(cache_key=key, placement_seed=seed, layout_signature="sig_new")
    assert _register_all(swapped, _layouts(4)) == [False, False, False, False]


def test_no_persistence_without_cache_key(tmp_path, monkeypatch):
    """Without a cache key the failures stay in-memory: a fresh instance sees nothing."""
    _isolate_cache_home(tmp_path, monkeypatch)

    writer = PhysicsSettleFailureCache(cache_key=None, placement_seed=5, layout_signature="sig")
    layouts = _layouts(3)
    _register_all(writer, layouts)
    writer.record_failures([layouts[1]])

    reader = PhysicsSettleFailureCache(cache_key=None, placement_seed=5, layout_signature="sig")
    assert _register_all(reader, _layouts(3)) == [False, False, False]


def test_no_persistence_without_seed(tmp_path, monkeypatch):
    """Without a fixed seed uids are not reproducible, so nothing is persisted."""
    _isolate_cache_home(tmp_path, monkeypatch)

    writer = PhysicsSettleFailureCache(cache_key="scene_e", placement_seed=None, layout_signature="sig")
    layouts = _layouts(3)
    _register_all(writer, layouts)
    writer.record_failures([layouts[1]])

    reader = PhysicsSettleFailureCache(cache_key="scene_e", placement_seed=None, layout_signature="sig")
    assert _register_all(reader, _layouts(3)) == [False, False, False]


# ----------------------------------------------------------------------------------------------------
# Layout signature
# ----------------------------------------------------------------------------------------------------


def _box(name, size=(0.2, 0.2, 0.2)):
    return DummyObject(name=name, bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=size))


def _scene():
    """A desk anchor with a box placed On it -- exercises an object-valued relation param (parent)."""
    desk = _box("desk", size=(1.0, 1.0, 0.1))
    desk.add_relation(IsAnchor())
    box = _box("box")
    box.add_relation(On(desk, clearance_m=0.01))
    return [desk, box]


def _sig(objects, num_envs=4, pool_size=8, max_placement_attempts=5):
    return compute_layout_signature(objects, num_envs, pool_size, max_placement_attempts)


def test_signature_is_deterministic():
    """Same composition hashes identically across calls (no object-id leakage via repr)."""
    assert _sig(_scene()) == _sig(_scene())


def test_signature_changes_on_object_swap():
    """Renaming/swapping an object changes the digest."""
    swapped = _scene()
    swapped[1].name = "mug"
    assert _sig(_scene()) != _sig(swapped)


def test_signature_changes_on_resize():
    """Resizing an object's bounding box changes the digest."""
    resized = _scene()
    resized[1].bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.5, 0.5, 0.5))
    assert _sig(_scene()) != _sig(resized)


def test_signature_changes_on_relation_param():
    """Changing a relation parameter changes the digest."""
    retuned = _scene()
    retuned[1].relations = [On(retuned[0], clearance_m=0.05)]
    assert _sig(_scene()) != _sig(retuned)


def test_signature_changes_on_object_order():
    """Reordering objects changes the digest, since enqueue order drives the uids."""
    reordered = list(reversed(_scene()))
    assert _sig(_scene()) != _sig(reordered)


def test_signature_changes_on_counts_and_retries():
    """num_envs, pool_size, and max_placement_attempts each feed the digest."""
    base = _sig(_scene())
    assert base != _sig(_scene(), num_envs=8)
    assert base != _sig(_scene(), pool_size=16)
    assert base != _sig(_scene(), max_placement_attempts=10)
