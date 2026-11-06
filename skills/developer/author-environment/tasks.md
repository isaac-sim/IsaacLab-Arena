# Tasks in environment graph YAML

Read this when choosing or composing the `task` block of an environment graph YAML. For the full
design, see [composite tasks](../../../docs/pages/concepts/task/concept_composite_tasks_design.rst)
and [progress tracking](../../../docs/pages/concepts/task/concept_progress_tracking_design.rst).

## Choose a task

`kind` is a registered task class name, and `params` are its constructor arguments, with assets
given by their node ids. Check the constructor in `isaaclab_arena/tasks/` for the parameters.

- `PickAndPlaceTask`: move `pick_up_object` into or onto `destination_location`. Most RoboLab tasks
  use it, such as [banana in bowl](../../../isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml).
  Its objects are marked `requires_reachability`, so placement only keeps layouts where the robot
  can reach them.
- `OpenDoorTask`, `CloseDoorTask`: open or close an articulated door, such as a fridge or cabinet.
- `PressButtonTask`, `TurnKnobTask`, `RotateRevoluteJointTask`: operate an articulated control.
- `NoTask`: a scene without success criteria.

Search `isaaclab_arena/tasks/` for others, such as sorting, lifting or placing upright, before
writing a new task.

## Compose several subtasks

`composition` combines the subtasks:

- `atomic`: exactly one subtask.
- `parallel`: the subtasks may finish in any order, as in
  [apple and yogurt in bowl](../../../isaaclab_arena_environments/robolab/tasks/apple_and_yogurt_in_bowl.yaml).
- `sequential`: the subtasks must finish in the listed order, as in
  [banana then Rubik's cube](../../../isaaclab_arena_environments/robolab/tasks/banana_then_rubiks_cube.yaml).

By default a subtask stays complete once it succeeds, even if its object later leaves the
destination. For example, with "put the banana and the orange in their bowls", knocking the banana
out while placing the orange would still count as success. To require every subtask's condition
to hold when the task ends, set the final state in the environment configuration override, one
entry per subtask in `subtasks` order:

```yaml
env_cfg_override:
  task_progress:
    desired_subtask_success_state: [true, true]
```

`true` requires the subtask to have completed and its condition to hold now, `false` requires it
to have completed and its condition to be false now, and `null` leaves it out of the success
check. The build prints the effective `subtasks_are_sequential` and `desired_subtask_success_state`;
read them there to confirm the setting was applied.
