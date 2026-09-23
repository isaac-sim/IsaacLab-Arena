# Rename ProgressObjective to CompletionCriteria

Follow-up to #1307. This note records the intended rename; it does not change the API yet.

## Why

Task success and progress now use the same requirements. `ProgressObjective` defines which
predicates must be satisfied, in what order, and which independent sequences must complete.
`CompletionCriteria` describes that responsibility more directly, for both tasks and subtasks.

The definition does not own sequence positions or temporal counters. Its runtime runner does.

## Scope

- Rename `ProgressObjective` to `CompletionCriteria` and its module to `completion_criteria.py`.
- Rename `ProgressObjectiveRunner` to `CompletionCriteriaRunner`.
- Align the completion-mode enum, state class, parameters, attributes, imports, tests, and examples.
- Keep `ProgressTracker`, `TaskSuccessTerm`, and `TaskTerminationCfg.success`.
- Update report fields and their readers together if they still use the old terminology.
  State any recorded-data format changes explicitly in the follow-up PR.
- Use `sequence` instead of the remaining `group` terminology for predicate sequences.
  Grouping runners by subtask is a different concept and does not need this rename.

Each `CompletionCriteria` instance defines a whole named set of requirements, not one predicate.
A list of these instances is a list of criteria sets; avoid calling each instance a `criterion`.
Choose the related parameter and report-field names consistently before implementation.

Update the `success_objectives` parameter in `TaskSuccessTerm` and `ArenaEnvBuilder` together;
Isaac Lab checks that configured parameters match the term's callable interface.

Existing recordings use keys such as `progress.objectives` and event `objective`.
Decide whether to keep those serialized keys or teach report readers to accept old recordings
when renaming them. Changing only the writer would leave existing report readers out of sync.

## Example

Here `settled`, `lifted`, and `placed` are already configured predicates.

Before:

```python
ProgressObjective(
    name="pick_and_place",
    predicate_sequence=[settled, lifted, placed],
)
```

After:

```python
CompletionCriteria(
    name="pick_and_place",
    predicate_sequence=[settled, lifted, placed],
)
```

## Boundaries and verification

This is a naming change, not a redesign of predicate evaluation, counters, resets, or scoring.
It does not introduce `PredicateCfg`. Do not add compatibility aliases solely to retain old names.

Check all task declarations, builder wiring, CAP diagnostics readers, recording, and evaluation
reports for old imports and field names. Run the progress-tracking, temporal-predicate, task-success,
composite-task, and reporting tests. Keep #1307's local sequence-name cleanup in #1307.
