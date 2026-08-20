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
