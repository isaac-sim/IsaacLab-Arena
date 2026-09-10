# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pool filtering and recycling contracts."""

from __future__ import annotations

import pytest

from isaaclab_arena.relations.pooled_object_placer import EnvLayoutPool, PooledObjectPlacer


class _Layout:
    """Tagged layout with an acceptance flag."""

    def __init__(self, tag: str, good: bool = True):
        self.tag = tag
        self.good = good


def _placer_with(*queues: list[_Layout]) -> PooledObjectPlacer:
    """Return a pool with explicit candidate layouts."""
    placer = PooledObjectPlacer.__new__(PooledObjectPlacer)
    placer._env_pools = [EnvLayoutPool(layouts=list(queue)) for queue in queues]
    placer._num_envs = len(queues)
    placer._recycle_layouts = False
    return placer


def _tags(placer: PooledObjectPlacer, env_id: int) -> list[str]:
    pool = placer._env_pools[env_id]
    return [layout.tag for layout in pool.layouts[pool.cursor :]]


def test_rejected_layouts_are_dropped():
    placer = _placer_with([_Layout("a"), _Layout("b", good=False), _Layout("c")])
    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good)

    assert (kept, rejected) == (2, 1)
    assert _tags(placer, 0) == ["a", "c"]


def test_every_layout_kept_when_all_pass():
    placer = _placer_with([_Layout("a"), _Layout("b")])
    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good)

    assert (kept, rejected) == (2, 0)
    assert _tags(placer, 0) == ["a", "b"]


def test_filter_rejects_empty_env_pool():
    placer = _placer_with([_Layout("a", good=False), _Layout("b", good=False)])
    with pytest.raises(AssertionError, match="Insufficient valid layouts"):
        placer.retain_layouts(lambda _env, layout: layout.good)
    assert _tags(placer, 0) == ["a", "b"]


@pytest.mark.parametrize("minimum", [2, 3])
def test_filter_enforces_minimum_survivors(minimum):
    placer = _placer_with([_Layout("a"), _Layout("b"), _Layout("bad", good=False)])
    if minimum == 2:
        assert placer.retain_layouts(lambda _env, layout: layout.good, minimum=minimum) == (2, 1)
        assert _tags(placer, 0) == ["a", "b"]
    else:
        with pytest.raises(AssertionError, match="Insufficient valid layouts"):
            placer.retain_layouts(lambda _env, layout: layout.good, minimum=minimum)
        assert _tags(placer, 0) == ["a", "b", "bad"]


def test_each_env_is_filtered_independently():
    placer = _placer_with(
        [_Layout("a0"), _Layout("b0", good=False)],
        [_Layout("a1"), _Layout("b1")],
    )
    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good)

    assert (kept, rejected) == (3, 1)
    assert _tags(placer, 0) == ["a0"]
    assert _tags(placer, 1) == ["a1", "b1"]


def test_predicate_receives_the_env_id():
    placer = _placer_with([_Layout("a0")], [_Layout("a1")])
    seen: list[int] = []

    def record(env_id: int, _layout) -> bool:
        seen.append(env_id)
        return True

    placer.retain_layouts(record)
    assert seen == [0, 1]


def test_consumed_layouts_are_not_reconsidered():
    placer = _placer_with([_Layout("used", good=False), _Layout("a"), _Layout("b", good=False)])
    placer._env_pools[0].next()

    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good)
    assert (kept, rejected) == (1, 1)
    assert _tags(placer, 0) == ["a"]


def _drawing_placer(*queues: list[_Layout]) -> PooledObjectPlacer:
    placer = _placer_with(*queues)
    placer._pool_size = sum(len(q) for q in queues)
    return placer


def test_recycling_rewinds_an_exhausted_queue():
    placer = _drawing_placer([_Layout("a"), _Layout("b")])
    placer.recycle_layouts = True

    drawn = [placer.sample_for_envs([0])[0].tag for _ in range(5)]
    assert drawn == ["a", "b", "a", "b", "a"]


def test_recycling_is_off_by_default():
    placer = _drawing_placer([_Layout("a")])
    assert placer.recycle_layouts is False


def test_recycling_rewinds_only_the_exhausted_envs():
    placer = _drawing_placer([_Layout("a0")], [_Layout("a1"), _Layout("b1")])
    placer.recycle_layouts = True

    first = placer.sample_for_envs([0, 1])
    assert [first[0].tag, first[1].tag] == ["a0", "a1"]
    second = placer.sample_for_envs([0, 1])
    assert [second[0].tag, second[1].tag] == ["a0", "b1"]


def test_recycling_survives_rejection_shrinking_the_queue():
    placer = _drawing_placer([_Layout("a"), _Layout("bad", good=False), _Layout("c")])
    placer.retain_layouts(lambda _env, layout: layout.good)
    placer.recycle_layouts = True

    drawn = [placer.sample_for_envs([0])[0].tag for _ in range(4)]
    assert drawn == ["a", "c", "a", "c"]


def test_include_consumed_reconsiders_a_layout_behind_the_cursor():
    placer = _placer_with([_Layout("used", good=False), _Layout("a"), _Layout("b", good=False)])
    placer._env_pools[0].next()

    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good, include_consumed=True)
    assert (kept, rejected) == (1, 2)
    assert _tags(placer, 0) == ["a"]


def test_filter_consumed_before_enabling_recycling():
    placer = _placer_with([_Layout("used", good=False), _Layout("a")])
    placer._env_pools[0].next()
    assert placer.recycle_layouts is False

    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good, include_consumed=True)
    assert (kept, rejected) == (1, 1)
    assert _tags(placer, 0) == ["a"]


def test_recycled_bulk_draw_rewinds_only_the_short_pools():
    placer = _drawing_placer(
        [_Layout("a0"), _Layout("b0")],
        [_Layout("a1"), _Layout("b1"), _Layout("c1")],
    )
    placer.recycle_layouts = True
    placer._env_pools[0].next()
    placer._env_pools[1].next()

    drawn = placer.sample_without_replacement(4)
    assert [layout.tag for layout in drawn] == ["b0", "b1", "a0", "c1"]


def test_recycled_bulk_draw_reaches_a_layout_the_batch_size_does_not_divide():
    placer = _drawing_placer([_Layout("a"), _Layout("b"), _Layout("c")])
    placer.recycle_layouts = True

    drawn = [[layout.tag for layout in placer.sample_without_replacement(2)] for _ in range(3)]
    assert drawn == [["a", "b"], ["c", "a"], ["b", "c"]]
    assert {tag for batch in drawn for tag in batch} == {"a", "b", "c"}


def test_recycled_bulk_draw_never_repeats_within_one_call():
    placer = _drawing_placer([_Layout("a"), _Layout("b"), _Layout("c")])
    placer.recycle_layouts = True
    placer._env_pools[0].next()

    for _ in range(4):
        batch = [layout.tag for layout in placer.sample_without_replacement(3)]
        assert sorted(batch) == ["a", "b", "c"], f"a single draw repeated a layout: {batch}"


def test_recycled_bulk_draw_refuses_a_request_larger_than_the_pool():
    placer = _drawing_placer([_Layout("a"), _Layout("b")])
    placer.recycle_layouts = True
    placer._env_pools[0].next()

    with pytest.raises(ValueError, match="holds only 2"):
        placer.sample_without_replacement(3)

    assert placer._env_pools[0].cursor == 1
    assert [layout.tag for layout in placer.sample_without_replacement(1)] == ["b"]


def test_filter_failure_preserves_all_queues():
    placer = _placer_with(
        [_Layout("a0"), _Layout("b0", good=False)],
        [_Layout("a1", good=False), _Layout("b1", good=False)],
    )
    with pytest.raises(AssertionError, match=r"env\(s\) \[1\]"):
        placer.retain_layouts(lambda _env, layout: layout.good)
    assert _tags(placer, 0) == ["a0", "b0"]
    assert _tags(placer, 1) == ["a1", "b1"]
