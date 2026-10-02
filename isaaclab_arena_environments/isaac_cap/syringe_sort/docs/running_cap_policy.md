# Run the CAP policy against Arena

Run the simulation in Arena Docker and the policy in your Isaac-cap checkout on
the host, using the same two-terminal workflow as gear and cable. All four
variants use the unchanged `syringe_packing/gap_perception` graph on current CAP
main; no task branch or local graph is required.

## Prerequisites

- A running Arena container with host networking and this checkout mounted at
  `/workspaces/isaaclab_arena`.
- Arena's optional `cap` dependencies installed. If missing, run
  `/isaac-sim/python.sh -m pip install -e '.[cap]'` in the container as your host user.
- A configured Isaac-cap checkout with its pinned submodules, GaP runtime, tool
  environments, and VLM credentials (`~/.config/gap/vlm.env`).
- Port 19000 available. Each graph process serves one environment and one episode.

## Terminal 1: inside Arena Docker

From `/workspaces/isaaclab_arena`, choose one command:

### Single

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/single_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

### Both

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/both_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

### Designated

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/designated_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

### Cluttered

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/cluttered_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Use `--viz none` for headless operation. Camera observations stay enabled either
way. Omit `--record_camera_video` when recordings are not needed.

Wait for `[CapPolicy] Environment ready; waiting for GaP at 127.0.0.1:19000`
before starting Terminal 2. Arena waits up to 180 seconds. To allow more startup
time, append `shared.policy.connect_timeout_s=600` to the Arena command.

## Terminal 2: in the CAP checkout on the host

The same command works for all four variants:

```bash
cd /path/to/Isaac-cap

GAP_PORT=19000 \
GAP_GRAPH=syringe_packing/gap_perception \
CAP_GAP_ROBOT_PROFILE=fr3 \
CAP_GAP_ARM_BASE_POSITION=-0.5,-0.1,0.912 \
CAP_GAP_CONTROL_FREQUENCY_HZ=50 \
CAP_GAP_HONOURS_ROLL=0 \
CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
CAP_GAP_CAMERA_NAME=overhead,eye_in_hand,agentview \
GAP_HAND_TO_FINGERTIP_Z=0.157 \
GAP_TCP_ROTATION_Z=0.7853981633974483 \
./arena_gap/scripts/run_gap_graph.sh
```

The 50 Hz control rate and roll setting are specific to syringe. Each Arena
experiment supplies CAP's workspace measurements in the robot-base frame
(`surface_z=-0.117`, `transport_z=0.19`) and the 0.8 rad normalization used by its
FR3 observation adapter. The environment itself uses the shared embodiment's
binary gripper action and 100/10 stiffness/damping.

## Results and repeat runs

Arena writes a timestamped directory under `outputs/`, containing
`arena_experiment_result.json`, `<variant>/episode_results_rebuild0.jsonl`,
`index.html`, and the requested camera videos. Check both the Run status and the
episode success flag. Graph completion or disconnection is not benchmark success;
on disconnect, Arena allows two seconds of settling before ending the episode.

Restart both commands for each trial. Arena can report success before CAP finishes
retracting; stop that graph with Ctrl-C after Arena exits. Change the placement by
appending `shared.environment_builder.placement_seed=43` to the Arena command.
For a different port, change both `GAP_PORT` and `shared.policy.port`.

Keep CAP's printed trace directory and both terminal logs when investigating a
failure. Check host networking and `ss -ltnp 'sport = :19000'` if connection fails.
The client supports one environment; increasing `num_envs` requires separate
policy processes and is not supported by these experiments.

## Compare single against native CAP

The placement solver changed between CAP's pinned Arena revision and this
checkout. A placement seed of 42 therefore does not produce the same layout in
both. `single_cap_native_seed42_experiment.yaml` uses Arena's cached-placement
API to replay the actual native seed-42 reset poses, with the same simulation
seed, cameras, robot configuration, and task limits. The accompanying JSONL
records the source CAP commit (`35e74598a7630597fb48d44bd71b94228c239737`).

For this comparison, run the following inside Arena Docker:

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/single_cap_native_seed42_experiment.yaml \
  --record_camera_video --viz kit
```

Then start the Terminal 2 CAP command above, adding `GAP_SEED=42` and
`PYTHONHASHSEED=42` to its environment. Stop that graph when Arena exits before
starting another trial.

For native CAP, run this in a separate trial from the CAP checkout:

```bash
cd /path/to/Isaac-cap
ARENA_EXPERIMENT_CONFIG=arena_gap/experiment_configs/tool_sort/vabar_tool_sorting_syringe_single_gap_baseline.yaml \
GAP_SEED=42 PYTHONHASHSEED=42 \
./arena_gap/scripts/run_arena_eval.sh --viz kit
```

CAP's launcher starts its policy automatically. It runs CAP's own benchmark
environment on its pinned Arena dependency, without importing this checkout.
Run the two trials sequentially to avoid competing for GPU resources. The
cached layout fixes the initial state; it does not make rendering, perception,
contact physics, or the asynchronous policy stream bitwise deterministic.

For an image-level comparison, load the cached layout in native CAP as well,
using its `environment_builder.placement_layouts_path` configuration field and
setting `placement_seed` to `null`. The native command above recomputes placement:
even when its resulting poses match, that initialization path can produce a
different first camera image. Loading the same cache in both runtimes avoids
that difference without changing either environment's randomization code.

The syringe experiments set the test client's `startup_render_steps` to zero.
The environment already refreshes cameras with five renders during reset,
matching native CAP. Rendering another five frames in the client changes the
initial image without moving the objects; in a cluttered-scene check, this
alone moved SAM's syringe confidence across the policy's detection threshold.
Camera initialization belongs to the environment reset for these experiments.

The first paired Single check succeeded in both runtimes, but a subsequent check
with identical initial images exposed a CAP policy instability. CAP's geometry
fitting adds random noise to the point cloud; the fallback grasp can select
opposite bounding-box axes and turn the wrist by approximately 180 degrees.
Replaying the same captured cloud through CAP reproduced that axis flip. Both
subsequent episodes failed the benchmark, with different wrist and carry paths.
Matching reset states therefore does not guarantee matching policy behavior,
and the initial successful pair is not a reliability claim.

The paired Both check used nearly identical grasp and carry plans and deposited
the first syringe in both runtimes. CAP then segmented the gripper as a syringe
near the aperture and rejected the placement, aborting before the second pickup.
Both benchmarks correctly reported the two-object task incomplete. This shared
policy verification failure is separate from environment configuration parity.

Designated succeeded in both runtimes with the red-cap syringe in the receiver
and the other syringe outside. Grasp estimates differed by 0.06 mm; both chose
the same carry strategy, with planned positions differing by at most 7.43 mm.
Cluttered aborted at initial syringe detection in both runtimes. That shared
failure checks the initial perception path, but does not validate the complete
six-syringe manipulation sequence. All four final pairs used identical cached
reset states and byte-identical first policy images. These are one-layout
diagnostic checks, not success-rate estimates.
