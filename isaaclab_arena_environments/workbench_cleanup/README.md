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
articulation loaders. Their scale, fixed base, articulation roots, and
openable or pressable joints are configured in `scene.yaml`; the same file
defines the workbench placement support. The workbench and socket set keep
their authored colliders, while hand tools use convex-hull contacts. Downloaded
models and generated USD wrappers stay in the user cache.

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
