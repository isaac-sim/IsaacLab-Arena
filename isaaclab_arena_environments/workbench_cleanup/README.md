# Workbench cleanup

`scene.yaml` is a shared scene include. Load `task_1.yaml` through
`task_4.yaml` with `ArenaEnvGraphSpec.from_yaml`. The embodiment is DROID.

Source credits: [NVIDIA SimReady assets](https://huggingface.co/datasets/nvidia/simready-assets),
[ForgeCAD](https://huggingface.co/datasets/ForgeCAD/simready-assets)
(`ratchet_socket_set`, CC BY 4.0; physics adapted here). Exact source revisions
are pinned in `scene.yaml`.

The pinned SimReady Central packages provide the blue workbench, white
two-drawer cabinet, ratchet socket set, oscilloscope, three screwdrivers, and two
sizes of combination wrench. Five `clutter_on` relations release the tools over
a fixed clutter area on the workbench.

The cabinet, socket set, and oscilloscope use the reusable SimReady
articulation loaders. Their fixed base, articulation roots, and openable or
pressable joints are configured in `scene.yaml`; the same file defines the
workbench placement support. The workbench and socket set keep their authored
colliders, while hand tools use convex-hull contacts. Downloaded models and
generated USD wrappers stay in the user cache.

The tasks are:

- Task 1: close the socket set.
- Task 2: open the top drawer of the white cabinet, put all five cluttered tools
  inside, and close the drawer.
- Task 3: switch the oscilloscope off.
- Task 4: open the top drawer of the white cabinet.

## Validated demos

The scripted force demo reports task success for each sequence:

Task 1:

![Task 1 closes the socket set](../../docs/images/workbench_cleanup/task_1/demo.gif)

Task 2:

![Task 2 loads and closes the white cabinet drawer](../../docs/images/workbench_cleanup/task_2/demo.gif)

Task 3:

![Task 3 switches off the oscilloscope](../../docs/images/workbench_cleanup/task_3/demo.gif)

Task 2 uses `ObjectInTask`: screwdrivers require 90% mesh-point containment,
while the two elongated wrenches use 50%. Tools must settle and have support
contact from the drawer or another task tool in the drawer. The pressed power
button represents the oscilloscope's off state; no electronics are simulated.

## Physical force-control recipes

Discover this clone's running container with the `dev-container` skill. Run as
the host user in `/workspaces/isaaclab_arena`:

```bash
HF_HUB_CACHE="$HOME/.cache/arena-workbench-hf" \
  ./submodules/IsaacLab/isaaclab.sh -p \
  isaaclab_arena/scripts/force_control_runner.py \
  --env_spec isaaclab_arena_environments/workbench_cleanup/task_1.yaml \
  --commands isaaclab_arena_environments/workbench_cleanup/task_1_force_control.jsonl \
  --output artifacts/workbench_cleanup/task_1 \
  --placement_seed 17 \
  --visualizer kit
```

Select the matching `task_N.yaml` and `task_N_force_control.jsonl` for Tasks 2
and 3. The generic runner also accepts registered Arena environments through
the normal environment subcommand.

Recipes use one JSON object per line. `setup_joint` records precede runtime
commands and add zero-gain effort actuators before the environment is created.
Runtime commands include `move`, `lift`, `joint`, `release`, `wait`, `inspect`,
`mark`, `check`, `wait_final`, and `finish`. A move uses a bounded physical
wrench and a joint command uses effort control; neither writes an object pose
during the task. Goals can be absolute or relative to another object's pose or
AABB center. For example:

```json
{"action":"joint","name":"white_cabinet","joint_name":"joint_part_01_joint_00","normalized_goal":0.95,"seconds":3,"gain":300,"damping":50,"limit":70}
```

Without `--follow_commands`, the runner executes the finite recipe and exits.
With `--follow_commands`, the input file may initially contain only its
`setup_joint` records; commands appended later are executed until a `finish`
record arrives.

Outputs go to `artifacts/workbench_cleanup/task_N/`: `demo.mp4`, subtask boundary
PNGs, `progress.json`, and `validation.json`. Videos and PNGs include a banner
with the current subtask and latched completion state. The process asserts final
task success unless `--allow_incomplete` is supplied. Use Fabric synchronization
when rendering these nested SimReady physics assets. For mouse dragging and
collision inspection instead of a reproducible recipe, use
`isaaclab_arena/scripts/environment_runner.py`.

## Robot-control recipes

The absolute-IK runner executes object-relative DROID pose, gripper, and
trajectory commands while checking the same task progress predicates:

```bash
./submodules/IsaacLab/isaaclab.sh -p \
  isaaclab_arena/scripts/robot_control_runner.py \
  --env_spec isaaclab_arena_environments/workbench_cleanup/task_1.yaml \
  --commands isaaclab_arena_environments/workbench_cleanup/task_1_robot_control.jsonl \
  --output artifacts/workbench_cleanup/task_1_robot \
  --placement_seed 17 \
  --visualizer kit
```

Select the matching `task_N.yaml` and `task_N_robot_control.jsonl` for Tasks
1–3. Pose targets may be expressed in the robot root, world, object, object
body, or AABB-center frame. A pose command can opt into cuRobo v2 planning with
`"planner":"curobo"`; the runner converts the planned joint path to dense
end-effector waypoints and executes them through the wrapped environment while
keeping task progress, recording, and termination behavior active.

### Task 1 robot validation

The Task 1 robot recipe passes with placement seed `17` (2026-10-31).
No command accepts a blocked or timed-out motion. The runner stops during the
closing sweep as soon as the native `close_door` predicate reports task success;
the remaining wait, check, and retreat commands are skipped. The final frame
and report capture episode `0` before Arena automatically resets it. The run
stops at control step 335 (22.33 simulated seconds). Only robot actions are
applied during execution.

![DROID closes the socket set](../../docs/images/workbench_cleanup/task_1_robot/demo.gif)

Artifacts: [start](../../docs/images/workbench_cleanup/task_1_robot/00_start.png),
[end](../../docs/images/workbench_cleanup/task_1_robot/00_end.png), and
[validation report](../../docs/images/workbench_cleanup/task_1_robot/validation.json).
The GIF plays at simulation speed. The runner also saves the full MP4 in its
output directory. The command above reproduces this run; other placement seeds
have not been validated.

The original first approach stopped 49.3 mm short of its target. Measured arm
joints followed their requested positions within 0.0003 radians, indicating
that increasing the wait or accepting the stall did not fix the IK target.
The new approach uses a closer offset from the lid's AABB center, then lowers
the fingertips to the upper portion of the lid and sweeps forward. The sweep
ends relative to the case root so its target does not follow the moving lid.
The shared scene starts the arm folded back above the oscilloscope to avoid
contact with it during reset.

Two runner/planner issues were also corrected:

- The runner stops immediately when a control step reports task success,
  including during interpolated moves, cuRobo paths, and waits. Arena's normal
  automatic reset remains enabled. An episode recorder captures the terminal
  frame and validation state before reset; later recipe commands are skipped.
- The cuRobo adapter applies the fixed URDF-to-USD end-effector offset on the
  tool side of the transform. A base-frame offset calibrated at one posture
  gave incorrect targets when the wrist rotated. Task 1 itself uses Arena's
  differential IK and does not require cuRobo.

Failed commands now save `validation.json`, including the error, measured and
target poses, joint positions, episode index, and native predicates, and close
the video writer. Video frame counts track simulation time. The focused
runner and planner tests cover rotated-wrist conversion, failure reporting,
success enforcement, stopping within a command, terminal-state recording across
automatic reset, and video timing.

Tasks 2 and 3 remain experimental robot recipes. Their force-control demos
above validate the task predicates; they do not establish successful robot
execution. A Task 3 probe with the corrected adapter presses the button but
stalls during both the approach and retreat; it is not a completed robot demo.
The same probe verifies the corrected cuRobo wrist transform against the
simulated wrist after motion (position difference below 0.4 micrometers).
Task 2 has not been revalidated with the new starting posture.
The optional cuRobo adapter currently plans without scene obstacle
geometry and executes Cartesian waypoints through differential IK, so it does
not guarantee collision-free scene motion or preservation of the planned
joint trajectory.
