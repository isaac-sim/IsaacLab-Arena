# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Condition allocation controls, independent of simulator timing."""

import json
from copy import deepcopy
from itertools import combinations

import pytest

from isaaclab_arena.evaluation.condition_episode_budget import ConditionEpisodeBudget, ObjectSetEpisodeBudgetCfg


def _budget(quotas=None):
    return ConditionEpisodeBudget(ObjectSetEpisodeBudgetCfg("fruit", quotas or {"A": 2, "B": 3}), ["A", "A", "B"])


@pytest.mark.parametrize("first", [[0, 1], [2], [0, 1, 2]])
def test_slots_share_a_quota_and_wait_for_slow_conditions(first):
    budget = _budget()
    assert budget.assign([2, 1, 0]) == [0, 1, 2]
    budget.complete(first)
    budget.assign(first)
    assert budget.started["A"] == 2
    assert not budget.is_complete
    while budget.active:
        finishing = list(budget.active)
        budget.complete(finishing)
        budget.assign(finishing)
    assert budget.is_complete
    assert budget.completed == {"A": 2, "B": 3}


@pytest.mark.parametrize("quotas", [{"A": 1, "B": 1}, {"A": 2, "B": 3}, {"A": 3, "B": 2}])
def test_every_completion_order_preserves_reservations(quotas):
    initial = _budget(quotas)
    initial.assign([0, 1, 2])
    pending = [initial]
    visited = set()
    terminal_count = 0
    while pending:
        budget = pending.pop()
        state = json.dumps(budget.snapshot(), sort_keys=True)
        if state in visited:
            continue
        visited.add(state)
        for member, requested in quotas.items():
            in_flight = sum(budget.members_by_env[i] == member for i in budget.active)
            assert budget.completed[member] + in_flight == budget.started[member] <= requested
        if budget.is_complete:
            terminal_count += 1
            assert budget.completed == quotas
            continue
        assert budget.active, "Unmet reachable quotas must not deadlock"
        for size in range(1, len(budget.active) + 1):
            for finishing in combinations(budget.active, size):
                next_budget = deepcopy(budget)
                next_budget.complete(list(finishing))
                next_budget.assign(list(finishing))
                pending.append(next_budget)
    assert terminal_count > 0


def test_interruption_retains_unstarted_and_in_flight_assignments(tmp_path):
    budget = _budget()
    path = tmp_path / "budget.json"
    budget.set_output_path(path)
    assert json.loads(path.read_text())["conditions"]["B"]["unstarted"] == 3
    budget.assign([0, 1, 2])
    budget.complete([0])
    result = budget.write("incomplete", "KeyboardInterrupt")
    assert result["conditions"]["A"] == {"requested": 2, "started": 2, "completed": 1, "in_flight": 1, "unstarted": 0}
    assert result["conditions"]["B"] == {"requested": 3, "started": 1, "completed": 0, "in_flight": 1, "unstarted": 2}
    assert [row["env_id"] for row in result["unfinished_assignments"]] == [1, 2]
    assert result == json.loads(path.read_text())
    with pytest.raises(AssertionError, match="unmet quotas"):
        budget.write("completed")
    assert json.loads(path.read_text()) == result


def test_invalid_batch_does_not_partially_mutate_accounting():
    budget = _budget()
    budget.assign([0, 1])
    before = budget.snapshot()
    for ids in ([0, 0], [0, 2]):
        with pytest.raises(AssertionError):
            budget.complete(ids)
        assert budget.snapshot() == before
    for ids in ([2, 2], [2, 4], [2, 0]):
        with pytest.raises(AssertionError):
            budget.assign(ids)
        assert budget.snapshot() == before


@pytest.mark.parametrize("quota", [0, -1, True, 1.5, "2"])
def test_rejects_invalid_quota(quota):
    with pytest.raises(AssertionError):
        ObjectSetEpisodeBudgetCfg("fruit", {"A": quota})


@pytest.mark.parametrize("members", [[], ["A"], ["A", "B", "C"]])
def test_rejects_unreachable_or_unbudgeted_conditions(members):
    with pytest.raises(AssertionError):
        ConditionEpisodeBudget(ObjectSetEpisodeBudgetCfg("fruit", {"A": 1, "B": 2}), members)


def test_configuration_and_snapshot_mutation_do_not_change_the_budget():
    cfg = ObjectSetEpisodeBudgetCfg("fruit", {"A": 1})
    definitions = {"A": {"index": 0, "spawn": {"scale": [1, 1, 1]}}}
    budget = ConditionEpisodeBudget(cfg, ["A"], definitions)
    cfg.episodes_per_member["A"] = 200
    definitions["A"]["spawn"]["scale"][0] = 20
    snapshot = budget.snapshot()
    snapshot["member_definitions"]["A"]["spawn"]["scale"][0] = 30
    assert budget.total == 1
    assert budget.snapshot()["member_definitions"]["A"]["spawn"]["scale"] == [1, 1, 1]
