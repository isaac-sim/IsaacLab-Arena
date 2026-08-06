# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for discarding pooled layouts that only prove unusable after simulating."""

from __future__ import annotations

from isaaclab_arena.relations.pooled_object_placer import EnvLayoutPool, PooledObjectPlacer


class _Layout:
    """Stands in for PlacementResult; rejection only reads what the predicate looks at."""

    def __init__(self, tag: str, good: bool = True):
        self.tag = tag
        self.good = good


def _placer_with(*queues: list[_Layout]) -> PooledObjectPlacer:
    """A placer whose env pools hold the given layouts, without solving anything."""
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


def test_env_keeps_its_rejects_when_too_few_survive():
    """An imperfect layout still beats an env having nothing to draw."""
    placer = _placer_with([_Layout("a", good=False), _Layout("b", good=False)])
    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good)

    assert rejected == 0, "an env with no passing layout must not be emptied"
    assert kept == 2
    assert _tags(placer, 0) == ["a", "b"]


def test_minimum_governs_whether_rejection_applies():
    passing = [_Layout(f"p{i}") for i in range(2)]
    placer = _placer_with([*passing, _Layout("bad", good=False)])

    # Two survivors clear a minimum of two, so the reject is dropped.
    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good, minimum=2)
    assert (kept, rejected) == (2, 1)

    # The same queue against a minimum of three keeps everything instead.
    placer = _placer_with([*passing, _Layout("bad", good=False)])
    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good, minimum=3)
    assert (kept, rejected) == (3, 0)


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
    """Only unread layouts matter; a consumed one is already behind the cursor."""
    placer = _placer_with([_Layout("used", good=False), _Layout("a"), _Layout("b", good=False)])
    placer._env_pools[0].next()

    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good)
    assert (kept, rejected) == (1, 1)
    assert _tags(placer, 0) == ["a"]


# ------------------------------------------------------------------ recycling


def _drawing_placer(*queues: list[_Layout]) -> PooledObjectPlacer:
    placer = _placer_with(*queues)
    placer._pool_size = sum(len(q) for q in queues)
    return placer


def test_recycling_rewinds_an_exhausted_queue():
    """A prepared pool must keep handing back prepared layouts, not solve new ones."""
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
    # Env 0 has run out and rewinds; env 1 still has an unread layout and advances.
    second = placer.sample_for_envs([0, 1])
    assert [second[0].tag, second[1].tag] == ["a0", "b1"]


def test_recycling_survives_rejection_shrinking_the_queue():
    placer = _drawing_placer([_Layout("a"), _Layout("bad", good=False), _Layout("c")])
    placer.retain_layouts(lambda _env, layout: layout.good)
    placer.recycle_layouts = True

    drawn = [placer.sample_for_envs([0])[0].tag for _ in range(4)]
    assert drawn == ["a", "c", "a", "c"], "recycling must cycle the layouts rejection left behind"


def test_include_consumed_reconsiders_a_layout_behind_the_cursor():
    """A caller that will rewind past the cursor must be able to judge what a rewind returns.

    Preparation consumes one layout before filtering the pool, and that is precisely the one a
    rewind hands back first, so leaving it unjudged replays a known-bad layout for the run.
    """
    placer = _placer_with([_Layout("used", good=False), _Layout("a"), _Layout("b", good=False)])
    placer._env_pools[0].next()

    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good, include_consumed=True)
    assert (kept, rejected) == (1, 2)
    assert _tags(placer, 0) == ["a"]


def test_reach_is_the_callers_to_declare_not_read_from_recycling():
    """The consumed layout is judged on the caller's say-so, whatever the recycling flag reads.

    The flag describes present sampling, not future reach: a caller may filter first and enable
    recycling afterwards, so reading it here would skip exactly the layout a rewind returns first.
    """
    placer = _placer_with([_Layout("used", good=False), _Layout("a")])
    placer._env_pools[0].next()
    assert placer.recycle_layouts is False, "recycling is still off at the moment the env filters"

    kept, rejected = placer.retain_layouts(lambda _env, layout: layout.good, include_consumed=True)
    assert (kept, rejected) == (1, 1)
    assert _tags(placer, 0) == ["a"]


# ------------------------------------------- recycling and bulk sampling


def test_recycled_bulk_draw_rewinds_only_the_short_pools():
    """One env running short must not rewind an env that still has unread layouts."""
    placer = _drawing_placer(
        [_Layout("a0"), _Layout("b0")],
        [_Layout("a1"), _Layout("b1"), _Layout("c1")],
    )
    placer.recycle_layouts = True
    placer._env_pools[0].next()
    placer._env_pools[1].next()

    # Env 0 is one short of the two-per-env round: it spends its unread b0 first and only then
    # wraps to a0. Env 1 has b1 and c1 unread, so it advances through both rather than
    # replaying a1.
    drawn = placer.sample_without_replacement(4)
    assert [layout.tag for layout in drawn] == ["b0", "b1", "a0", "c1"]


def test_recycled_bulk_draw_reaches_a_layout_the_batch_size_does_not_divide():
    """A batch that does not divide the pool must still reach every layout.

    Rewinding before the draw strands whatever sits behind the cursor: three layouts drawn two
    at a time would return the first two forever and never reach the third.
    """
    placer = _drawing_placer([_Layout("a"), _Layout("b"), _Layout("c")])
    placer.recycle_layouts = True

    drawn = [[layout.tag for layout in placer.sample_without_replacement(2)] for _ in range(3)]
    assert drawn == [["a", "b"], ["c", "a"], ["b", "c"]]
    assert {tag for batch in drawn for tag in batch} == {"a", "b", "c"}


def test_recycled_bulk_draw_never_repeats_within_one_call():
    """Wrapping mid-draw is bounded by the capacity precheck, so a single call stays distinct."""
    placer = _drawing_placer([_Layout("a"), _Layout("b"), _Layout("c")])
    placer.recycle_layouts = True
    placer._env_pools[0].next()

    for _ in range(4):
        batch = [layout.tag for layout in placer.sample_without_replacement(3)]
        assert sorted(batch) == ["a", "b", "c"], f"a single draw repeated a layout: {batch}"


def test_recycled_bulk_draw_refuses_a_request_larger_than_the_pool():
    """Without replacement and beyond the prepared set conflict, so the request is refused."""
    placer = _drawing_placer([_Layout("a"), _Layout("b")])
    placer.recycle_layouts = True
    placer._env_pools[0].next()

    try:
        placer.sample_without_replacement(3)
    except ValueError as error:
        assert "holds only 2" in str(error)
    else:
        raise AssertionError("an oversized recycled request must be refused")

    # Refused before anything moved: the half-served round must not have advanced the cursor.
    assert placer._env_pools[0].cursor == 1
    assert [layout.tag for layout in placer.sample_without_replacement(1)] == ["b"]
