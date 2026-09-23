# Completion criteria naming

The rename in #1316 uses `CompletionCriteria` for the named requirements shared by task
success and progress tracking. The step-update simplification in #1307 is separate.

## Responsibility

`CompletionCriteria` defines which predicates must be satisfied, in what order, and which
independent sequences must complete. Each instance defines a whole named set of requirements,
not one predicate. A list of these instances is a list of criteria sets.

`CompletionCriteriaRunner` evaluates those requirements and owns their runtime sequence
positions and temporal counters. `ProgressTracker` advances the runners, combines their
results, and reports progress. `TaskSuccessTerm` uses that tracker to determine task success.

## Python API

| Previous name | Current name |
| --- | --- |
| `progress_tracking.progress_objective` | `progress_tracking.completion_criteria` |
| `ProgressObjective` | `CompletionCriteria` |
| `ProgressObjectiveRunner` | `CompletionCriteriaRunner` |
| `ProgressObjectiveCompletionMode` | `SequenceCompletionMode` |
| `ProgressObjectiveState` | `CompletionCriteriaState` |
| `TaskSuccessTerm.success_objectives` parameter | `TaskSuccessTerm.success_criteria` parameter |
| Runner `progress_objective` | Runner `completion_criteria` |
| `ProgressTracker.progress_objectives` | `ProgressTracker.criteria_sets` |
| `ProgressState.progress_objectives` | `ProgressState.criteria_by_name` |
| `get_predicate(objective_name=...)` | `get_predicate(criteria_name=...)` |
| `group_names` | `sequence_names` |
| `DEFAULT_GROUP_NAME` | `DEFAULT_SEQUENCE_NAME` |
| `get_chain()` | `get_sequence()` |
| Event `progress_objective` | Event `criteria_name` |
| Event `group` | Event `sequence` |
| State `completed_groups` / `total_groups` | State `completed_sequences` / `total_sequences` |

`ArenaEnvBuilder` passes `success_criteria` to `TaskSuccessTerm`; Isaac Lab requires the
configured parameter name to match the term's callable interface. `TaskTerminationCfg.success`
still holds the list of required criteria sets. `ProgressTracker` and `TaskSuccessTerm` keep
their names.

The unnamed predicate sequence is now called `default_sequence`. Named criteria and sequences
retain their identifiers, including names such as `pick_and_place` and the composite-task
prefix `subtask_<index>/`. Grouping runners by subtask is a separate concept and is unchanged.

## Recording schema

Episode JSONL records and their report readers use the renamed fields together:

| Previous field or value | Current field or value |
| --- | --- |
| `progress.objectives` | `progress.criteria_by_name` |
| `progress.events[].objective` | `progress.events[].criteria_name` |
| `progress.events[].group` | `progress.events[].sequence` |
| `completed_groups` / `total_groups` | `completed_sequences` / `total_sequences` |
| Unnamed sequence `default_group` | Unnamed sequence `default_sequence` |

There are no aliases or readers for the previous schema. Older recordings require conversion
or regeneration before use with the current report tools. Conversion must rename the fields
above, including the unnamed sequence in `active_predicates` and event entries.

## Example

Here `settled`, `lifted`, and `placed` are already configured predicates:

```python
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria

criteria = CompletionCriteria(
    name="pick_and_place",
    predicate_sequence=[settled, lifted, placed],
)
```

## Scope

The rename preserves predicate evaluation, counters, resets, scoring, and task-success
behavior. It does not introduce `PredicateCfg`, compatibility aliases, or the separate
step-update simplification from #1307. Task declarations, builder wiring, CAP diagnostics,
recording, report readers, tests, and examples use the new names together.
