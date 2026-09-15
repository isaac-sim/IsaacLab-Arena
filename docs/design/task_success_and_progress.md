# Task success and progress from one definition

## Bigger picture

A task should define successful completion once. Progress reporting should show how far an
environment has reached toward that same definition, and success should be reported when the
required objectives are complete.

For `PickAndPlaceTask`, that means observing these predicates in order:

```text
objects_settled → object_is_above_height → object_on_destination
```

The old API configures progress tracking and success separately. Progress can say that lifting
has not happened while the success check says that placement is already successful. Reusing the
same predicate functions does not connect the history tracked by one system to the other.

This proposal makes the responsibilities explicit:

- `TaskTerminationCfg` (New) declares success objectives, failure conditions, and the episode time limit.
- `ProgressTracker` remembers which required predicates have been satisfied.
- `TaskSuccessTerm` (New) connects `ProgressTracker` to Isaac Lab's success checks and episode resets.
- `ProgressTrackingRecorder` (Simplified) reports the same progress without updating it independently.

More importantly, this also gives future stateful predicates a shared update and reset path. For a condition such
as "remain stable for 10 steps", `ProgressObjectiveRunner` could count qualifying steps while that
predicate is active. The counter could follow those same calls, so progress
reporting and success would use the same state.

With separate progress and success checks, using that condition in both places would require
much more complicated coordination: which check updates the shared counter, and which resets it?


## What the old task API makes difficult

The examples below compare the same predicates, failure condition, and time limit. Both use
`settled`, `lifted`, and `placed` for configured predicate callables, and `object_dropped` for the
configured failure term. These names are shorthand for the examples, not new task attributes or
APIs. Their unchanged argument setup is shown once after the comparison. Imports, constructors,
and unrelated task methods are omitted from both versions.

### Before: separate success, progress, and time-limit definitions

These excerpts collect the old definitions in one place. `PickAndPlaceTask` creates and caches
its `TerminationsCfg` during construction. `TaskBase` supplies the inherited time-limit getter;
task authors do not need to implement that getter themselves.

```python
@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(
        func=mdp_isaac_lab.time_out,
        time_out=True,
    )
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


# PickAndPlaceTask methods:
def make_termination_cfg(self):
    return TerminationsCfg(
        success=TerminationTermCfg(func=placed),
        object_dropped=object_dropped,
    )


def get_termination_cfg(self):
    return self.termination_cfg


def get_progress_objectives(self) -> list[ProgressObjective]:
    return [
        ProgressObjective(
            name="pick_and_place",
            predicate_groups=[settled, lifted, placed],
        ),
    ]


# Inherited from TaskBase:
def get_episode_length_s(self) -> float:
    return self.episode_length_s
```

`ArenaEnvBuilder` reads termination terms, progress objectives, and the time limit through three
separate getters. A reader must connect these definitions to understand what the task tracks
and what actually ends an episode.

`TerminationManager` evaluates `placed` without asking `ProgressTracker` whether the earlier
predicates were satisfied. Placement can therefore end the episode before the tracked lift
predicate has been satisfied. Even reusing the same configured callable, as shown here, does not
make success depend on the tracked sequence.

Progress updates and resets are spread out too. `ProgressTrackingRecorder` advances
`ProgressTracker`, while a separately registered `progress_tracking_reset_func()` clears its
state when environments reset.

## After: one task termination definition

`TaskBase.get_termination_cfg()` now returns `TaskTerminationCfg`. A task places its required
`ProgressObjective` definitions directly in `success`, alongside its failures and timeout.

Using the same configured predicates and failure term, `PickAndPlaceTask` now declares all
termination criteria in one method:

```python
def get_termination_cfg(self) -> TaskTerminationCfg:
    return TaskTerminationCfg(
        timeout_s=self.episode_length_s,
        success=[
            ProgressObjective(
                name="pick_and_place",
                sequence=[settled, lifted, placed],
            ),
        ],
        failures={"object_dropped": object_dropped},
    )
```

The task no longer needs its own `TerminationsCfg` class or a separate
`get_progress_objectives()` method.
`ArenaEnvBuilder` now reads the time limit from `TaskTerminationCfg.timeout_s` alongside success
and failure conditions.

The important simplification is one definition for progress and success.
`TaskSuccessTerm` reports success only after `settled`, `lifted`, and `placed`
have been satisfied in order. Placement alone can no longer bypass the earlier predicates.

### Predicate and failure setup: unchanged in both examples

Both versions need the same task-specific arguments. The shorthand above represents this setup:

```python
settled = partial(
    objects_settled,
    object_names=[self.pick_up_object.name],
)
lifted = partial(
    object_is_above_height,
    object_name=self.pick_up_object.name,
    use_settled_state=True,
)
placed = partial(
    object_on_destination,
    object_cfg=SceneEntityCfg(self.pick_up_object.name),
    destination_cfg=SceneEntityCfg(self.destination_location.name),
    contact_sensor_cfg=SceneEntityCfg(self.contact_sensor_name),
    force_threshold=self.force_threshold,
    velocity_threshold=self.velocity_threshold,
    support_cone_half_angle_rad=self.support_cone_half_angle_rad,
)
object_dropped = TerminationTermCfg(
    func=mdp_isaac_lab.root_height_below_minimum,
    params={
        "minimum_height": self.background_scene.object_min_z,
        "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
    },
)
```

### What the configuration means

- `success`: every listed `ProgressObjective` must complete for the task to succeed.
  An empty list disables success termination and does not create a `ProgressTracker`.
- `sequence`: predicates must be satisfied in order. Each sequence advances by at most one
  predicate per environment on each `ProgressTracker.step()` call, which `TaskSuccessTerm` makes
  once during a normal environment step. Predicates return one Boolean result per environment.
- `failures`: any configured failure can end the episode, even if success objectives are incomplete.
- `timeout_s`: a finite, positive episode time limit. `ArenaEnvBuilder` uses it to set Isaac Lab's
  episode length and installs the timeout term.

Sequence completion records history. Once the lift predicate has been satisfied, it does not have
to stay true during placement. Completed predicates are not continuously rechecked, and there is
no requirement yet that a predicate remain true for several steps.

For independent sequences, `ProgressObjective` still accepts
`predicate_groups={"group_name": [...]}` instead of `sequence`. Its `logical` setting determines
whether all groups, any group, or a chosen number of groups must finish. These are alternative
ways to define one objective; supply exactly one of `sequence` or `predicate_groups`.
Every objective listed in `TaskTerminationCfg.success` remains required.
Named groups track their own sequence positions, but observe the same environment. Scores affect
progress reporting only; success uses completion flags, not a score threshold.

Success is not the only reason an episode can end. Failure and timeout remain separate
`TerminationManager` terms. This change does not introduce a priority rule for cases where success
and failure are both true on the same step.

## Where responsibilities move

`ProgressObjective` remains a definition.
`ProgressObjectiveRunner` still owns the current predicate  index, score, and completion flags for each environment.
`ProgressTracker` holds those runners and their progress events.
--> The change is who creates, updates, and resets `ProgressTracker`.

| Responsibility | Before | After |
| --- | --- | --- |
| Define required success predicates | Task success function, separate from tracked objectives | `TaskTerminationCfg.success` |
| Create `ProgressTracker` | `_ensure_progress_tracker()`, called when first needed by recording or reset | `TaskSuccessTerm.__init__()` |
| Advance `ProgressTracker` | `ProgressTrackingRecorder.record_post_step()` | `TaskSuccessTerm.__call__()`, during termination evaluation |
| Report task success | Separately configured success function | `TaskSuccessTerm` returns `ProgressTracker.is_complete()` |
| Trigger the progress reset | `progress_tracking_reset_func()`, registered as a reset event | `TaskSuccessTerm.reset()`, called by `TerminationManager` |
| Publish progress | `ProgressTrackingRecorder` | `ProgressTrackingRecorder`, now only reading progress |

### ArenaEnvBuilder connects the task to Isaac Lab

When a task defines success objectives, `ArenaEnvBuilder._build_termination_manager_cfg()`
automatically adds this term to the Isaac Lab configuration:

```python
success_objectives = task_termination_cfg.success
if success_objectives:
    termination_terms["success"] = TerminationTermCfg(
        func=TaskSuccessTerm,
        params={"success_objectives": success_objectives},
    )
```

`TaskSuccessTerm` inherits from `ManagerTermBase`. `TerminationManager` constructs the term,
calls it during termination evaluation, and calls its `reset()` during environment resets.
`TaskSuccessTerm` creates and owns one `ProgressTracker` for the vectorized environment.

Failure predicates remain directly in `TerminationManager`; they do not become ordered success
predicates. Scene and embodiment termination conditions are still included, but neither component
may supply its own success term. The task supplies the success objectives.


### Resetting the state machine is preserved

The call to `ProgressTracker.reset()` moves from `progress_tracking_reset_func()`, registered
as an `EventManager` reset callback, to `TaskSuccessTerm.reset()`, called by `TerminationManager`.

```text
Environment reset for env_ids
  → TerminationManager.reset(env_ids)
      → TaskSuccessTerm.reset(env_ids)
          → ProgressTracker.reset(env_ids)
              → ProgressObjectiveRunner.reset(env_ids)
```

`TerminationManager` resets its registered class-based terms regardless of which condition ended
the episode. This covers success, failure, timeout, and manual environment resets.

`ProgressObjectiveRunner.reset()` still clears predicate indices, scores, and completion flags.
`ProgressTracker.reset()` also clears progress events. `TaskSuccessTerm.reset()` preserves the
existing reset of recorded initial resting positions used by the pick-and-place predicates.
Only the restarting environments are affected; other parallel environments keep their progress.

`ManagerTermBase` does not discover and reset nested objects automatically. `TaskSuccessTerm`
explicitly forwards the call to `ProgressTracker`, which forwards it to each
`ProgressObjectiveRunner`. We do not need an additional `PredicateGroup` manager-term wrapper
or a separately configured progress-reset event for this connection.

## Public API changes and migration

Task authors need to:

1. Return `TaskTerminationCfg` from `get_termination_cfg()`.
2. Move required objectives from `get_progress_objectives()` into `TaskTerminationCfg.success`.
   A task with only a single success predicate can use a one-predicate `sequence`.
3. Replace the old flat `predicate_groups=[...]` form with `sequence=[...]`. Named independent
   groups keep the dictionary form, with a nonempty list for each group. Wrap single predicates
   in a list: `predicate_groups=predicate` becomes `sequence=[predicate]`, and
   `predicate_groups={"group": predicate}` becomes `predicate_groups={"group": [predicate]}`.
4. Put failure terms in `failures` and the episode time limit in `timeout_s`. Remove the task's
   old success/timeout configuration and any cached `TerminationsCfg` that it no longer needs.

`TaskBase.get_progress_objectives()` is removed. `get_episode_length_s()` still exists, but
`ArenaEnvBuilder` now takes the episode time limit from `TaskTerminationCfg.timeout_s`.
An empty `success` list is supported for environments such as `NoTask`; it does not provide
independent progress-only tracking.


`ArenaEnvBuilder` combines named termination terms with task terms taking precedence over
embodiment terms, and embodiment terms over scene terms. The builder owns the `success` and
`time_out` names.


## Code references

- [TaskTerminationCfg](../../isaaclab_arena/tasks/task_termination_cfg.py)
- [PickAndPlaceTask](../../isaaclab_arena/tasks/pick_and_place_task.py)
- [ProgressObjective](../../isaaclab_arena/progress_tracking/progress_objective.py)
- [TaskSuccessTerm](../../isaaclab_arena/progress_tracking/task_success.py)
- [ProgressTracker, ProgressObjectiveRunner, and ProgressTrackingRecorder](../../isaaclab_arena/progress_tracking/progress_tracker.py)
- [ArenaEnvBuilder](../../isaaclab_arena/environments/arena_env_builder.py)
- [Success and reset regression tests](../../isaaclab_arena/tests/test_task_success_from_progress.py)
