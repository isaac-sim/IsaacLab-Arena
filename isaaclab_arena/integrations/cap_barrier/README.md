# CAP Arena barrier smoke

This directory is the Kit-side half of the CAP ROS v0.1 shared-memory barrier. The
normative ABI remains `cap_backend_arena/protocol.hpp` in `isaac_ros_cap`; the checked-in
JSON and binary fixtures under `isaaclab_arena/tests/test_data/cap_barrier` are generated
by its test-only `cap_arena_abi_golden_dump` target.

The original runner remains a deliberately fixed, non-blocking integration diagnostic. It is
designed to exercise one B=1 DROID environment at a 200 Hz declared base:

1. bootstrap generation 1 with FENCE frames while Kit physics is frozen;
2. while the arm remains in hold, observe the physical finger move open-to-close-to-open;
3. run a short joint-streaming trajectory in lockstep;
4. stage an Arena reset without a physics step and attach generation 2;
5. emit the reset fence, resume in hold, and shut down cleanly.

Historical Franka smoke results do not transfer to this embodiment. Any claimed DROID diagnostic
pass must be re-earned after the producer revision is pinned, including an observed finger
transition so an already-open initial state cannot make the gripper path pass vacuously. This fixed
runner is not the capability-advertisement gate; the production three-process smoke below is the
authoritative gate.

The fixed CAP builder removes the Gaussian offset from DROID's generic joint reset while retaining
the event's deterministic state write. The event reuses a Franka helper which preserves only the
final two articulation joints; on DROID those are mimic joints, not the commanded `finger_joint`.
Each generation therefore starts at the deterministic open endpoint rather than an unsupported
intermediate position.

For each frame, Kit publishes joint state and waits for the matching controller command.
It keeps the shared phase at `COMMAND_READY` while applying a PHYSICS command and calling
`env.step()` exactly once. Only after the step returns does it release `AWAIT_STATE`.
FENCE commands are discarded without calling `env.step()`. This makes `AWAIT_STATE` a
real simulator-quiescence point for generation changes.

The `arena_droid_b1` joint roster is exactly the seven `fr3_joint*` ABI joints followed by the
virtual `robotiq_85_left_knuckle_joint`. The producer maps those names to Arena's seven ordered
`panda_joint*` joints followed by `finger_joint`; it does not rely on articulation indices because
the DROID USD also contains mimic joints. Slot 7 remains in radians (`0` open, `pi/4` closed) in
both state and command frames. Arena's action is binary closedness (`0` open, `1` closed), so the
producer accepts only the two endpoint bands with the `arena_droid_b1` profile tolerance of
`0.01 rad` and rejects nonfinite or intermediate commands before stepping physics. Until shared
profile assembly lands, the ROS gripper relay configuration must mirror that tolerance explicitly.
The smoke brackets each commanded close and open transition with synchronized monotonic timestamps,
requires physical slot 7 to cross the half-closed position and reach the requested endpoint within
the declared `2 s` gripper bound. Commanded arm slots must remain exactly at their held values; the
physical arm reaction is accumulated across the complete open-close-open transition. The calibrated
envelope is `0.00016736984252929688 rad` (`0.00959 deg`), and the hard physical-isolation limit is
exactly twice it, `0.00033473968505859375 rad` (`0.01918 deg`). A peak at or below the envelope
passes; a peak above the envelope through the hard limit is calibration drift; and a peak above the
hard limit is a physical-isolation failure. Command-frame echo alone cannot satisfy this proof.

The envelope comes from ten uncensored full-transition GPU runs on 2026-07-17 at producer commit
`9c89f96ff39777ab2f0c0ba73b2048fdd8e7ce31`, using `arena_droid_b1`,
`CAP-Barrier-DROID-B1-v0`, `droid_abs_joint_pos`, the real `robotiq_gripper_controller`,
`sim.dt=0.005`, `decimation=1`, `num_envs=1`, Isaac Sim `6.0.0.1`, Isaac Lab `3.0.0b2`, and
`cuda:0` float32 execution on an NVIDIA GeForce RTX 4090 with driver `580.159.04`. Their maxima
formed three observed float32 levels:
`0.0001494884490966797 rad` in six runs,
`0.0001621246337890625 rad` in two, and `0.00016736984252929688 rad` in two. Command delta was zero
in every run; every physical maximum occurred while reopening, at samples 182 through 360. The
histogram does not prove that a fourth level is impossible, so any new maximum above the envelope
deliberately fails. With unchanged configuration, extend the calibration and adopt the new maximum.
Changes to source or dependency pins, scene, embodiment, controller or gains, simulator or physics
settings, device or dtype, hardware, timing, or producer measurement invalidate this calibration and
require the complete config-scoped calibration again.

The ABI's legacy-named `wait_interrupted` field is the atomic serviceability/reservation word. Its
layout is unchanged, but the four values and transitions are normative:

| Value | State | Legal transition | Writer |
| --- | --- | --- | --- |
| `0` | `SERVICEABLE` | `0 -> 2` | Producer reserves one complete exchange. |
| `0` | `SERVICEABLE` | `0 -> 1` | Sidecar deactivation excludes the next producer. |
| `1` | `INTERRUPTED` | `1 -> 0` | Sidecar activation clears the fence after its tick thread is live. |
| `2` | `PRODUCER_RESERVED` | `2 -> 0` | Producer completes, or releases after first faulting an abnormal abandonment. |
| `2` | `PRODUCER_RESERVED` | `2 -> 3` | Sidecar deactivation records pending reset intent. |
| `3` | `PRODUCER_RESERVED_INTERRUPT_PENDING` | `3 -> 1` | Producer completes, or releases after first faulting an abnormal abandonment. |

Kit attaches only when generation and `AWAIT_STATE` match and the word is `SERVICEABLE`. Before
copying each state frame it atomically reserves `0 -> 2`, holds the reservation through command
application and physics, then releases `2 -> 0` or `3 -> 1`. Generation reset is permitted only in
`INTERRUPTED`, which prevents the owner from clearing shared frame storage while Kit accesses it.
An exception or orderly client close while reserved latches `BARRIER_STATE_VIOLATION` before that
release. A hard process death cannot fault or release, so the owner detects it with its bounded
interrupt deadline and requires stale-object recovery. Initial zero remains intentional because a
newly created bootstrap barrier has no retiring producer.

## Shared-memory visibility

The standard Isaac ROS development container already runs with host IPC. Verify before
the smoke if the container invocation has changed:

```bash
docker inspect -f '{{.HostConfig.IpcMode}}' isaac_ros_dev_container
```

The expected output is `host`; no additional `~/.isaac_ros_dev-dockerargs` entry is
needed in the canonical setup.

## Fixed diagnostic

Build `isaac_ros_cap` with `BUILD_TESTING=ON`, then start the test-only
`cap_smoke_orchestrator` inside `isaac_ros_dev_container`. Wait until it prints
`CAP_SMOKE_READY_FOR_KIT`; the shared object exists before the sidecar is active, so
starting Kit on shared-memory existence alone is raceable.

From the Arena worktree on the host:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
PYTHONPATH=/home/rafael/Projects/arena-cap-barrier \
/home/rafael/Projects/Isaac-cap/external/IsaacLab-Arena/.venv/bin/python \
  isaaclab_arena/scripts/run_cap_barrier_smoke.py --viz none --device cuda:0
```

For this diagnostic to be green, the Kit side must emit
`CAP_SMOKE_KIT_GRIPPER_TRANSITION_OK` before `CAP_SMOKE_KIT_DONE`, and the ROS side must finish with
`CAP_SMOKE_ORCHESTRATOR_DONE`. An orchestrator-only completion is not a diagnostic pass. The
diagnostic is useful for protocol development but is non-blocking and must not be cited as
capability-advertisement evidence.

## Production control-plane advertisement gate

The production runner is the authoritative DROID capability-advertisement gate. It leaves the reset
instant to the ROS Session Manager action and uses three processes: the custom in-process ROS
composition container, the ROS-free Kit producer, and a test-only typed ROS client. Build
`isaac_ros_cap` with `BUILD_TESTING=ON`; the client is intentionally not installed and must be run
from the colcon build tree. The canonical bounded supervisor and cleanup-attestation command is
`scripts/run_cap_barrier_production_e2e.py` in the pinned Isaac-cap repository; the manual commands
below are for diagnosis only.

Cold-start Kit first in terminal 1 on the host:

```bash
cd /home/rafael/Projects/arena-cap-barrier
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
PYTHONPATH="$PWD" \
/home/rafael/Projects/Isaac-cap/external/IsaacLab-Arena/.venv/bin/python \
  isaaclab_arena/scripts/run_cap_barrier_production_smoke.py --viz none --device cuda:0
```

Kit constructs the complete DROID environment before emitting `CAP_PRODUCTION_KIT_ENV_READY` and
touching shared memory. It then has one 300-second startup-only rendezvous deadline spanning shared
memory creation, initialization, generation discovery, and the sidecar becoming serviceable. The
deadline is not the operational frame timeout: command frames and generation fences retain their
30-second safety bound.

After `CAP_PRODUCTION_KIT_ENV_READY`, start the control plane in terminal 2 inside the Isaac ROS
development container. Replace the build and install bases below if colcon used different paths:

```bash
docker exec -it -u admin isaac_ros_dev_container bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /workspaces/isaac_ros-dev/ros_ws/install/setup.bash && \
   ros2 launch cap_backend_arena arena_control_plane.launch.py'
```

The bootstrap generation is published interrupted until lifecycle activation makes the sidecar
serviceable. Seeing the shared-memory name or generation is therefore not permission to reserve the
barrier; the Kit producer retries within the one startup deadline. This closes both cold-start race
orders without extending the operational generation-fence timeout.

After `CAP_CONTROL_PLANE_ACTIVE generation=1`, start the typed test client in terminal 3. It waits
for Kit-driven bootstrap and the production
endpoints, drives a guarded close then open gripper operation, releases that lease, then acquires
`joint_streaming`, publishes through the validating relay, and invokes the real reset action:

```bash
docker exec -it -u admin isaac_ros_dev_container bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /workspaces/isaac_ros-dev/ros_ws/install/setup.bash && \
   /workspaces/isaac_ros-dev/ros_ws/build/cap_backend_arena/cap_production_smoke_client'
```

After the generation-1 bootstrap fence, Kit samples the owner generation and consumer-serviceability
latch before every next PHYSICS publication. It stops generation-1 publication as soon as the reset
becomes observable, waits for exact generation 2, resets DROID without advancing physics, attaches
only after the sidecar is serviceable, and emits the generation-2 fence. The
`CAP_PRODUCTION_KIT_GENERATION_2_DETECTED` marker includes the actual generation-1 physics count.
After the client and Kit markers appear, send Ctrl-C to terminal 2. Success requires
`CAP_CONTROL_PLANE_ACTIVE generation=1`,
`CAP_PRODUCTION_CLIENT_GRIPPER_TRANSITION_OK`,
`CAP_PRODUCTION_CLIENT_RESET_ACTION_OK`,
`CAP_PRODUCTION_KIT_GRIPPER_TRANSITION_OK`,
`CAP_PRODUCTION_KIT_GENERATION_2_DETECTED`,
`CAP_PRODUCTION_KIT_HOLD_RESUME_OK`,
`CAP_PRODUCTION_KIT_DONE`, zero Kit/client exits, a clean control-plane shutdown, and positive local
and remote process-group death attestations from the canonical supervisor.

## Deliberate boundaries

- The fixed controller timing roster is smoke data, not a CR-21 roster/discovery answer.
- `applied_torque` is carried as smoke-level effort data; real/sim World State effort
  parity remains open.
- Arm-command finite-value policy and safety fallback composition remain CR-22 work; the DROID
  gripper endpoint is already finite and discrete at this producer boundary.
- The production control plane owns reset admission and fence closure; the Kit producer remains
  ROS-free and observes only the CR-05 shared-memory contract.

## Perception producer (Phase 3)

`perception_producer.py` is the ROS-free Kit side of CAP perception. `extract_camera_frame`
reads the `exterior_cam` RGB-D output, intrinsics, and world pose from an already-stepped
environment (RTX rendering is triggered by that data access, so the barrier pays no render
cost until a frame is wanted). `PerceptionFrameProducer` owns a background gRPC
client-streaming thread with a single-slot latest-frame mailbox: `offer` never blocks the
physics loop and drops the previous unsent frame rather than back-pressuring lockstep. The
thread reconnects and restarts the stream on failure, requeues the latest real frame across a
restart, and never fabricates a frame. The frozen transport is `CapPerception.PublishFrames`
from the ROS module's `cap_perception.proto`.

The pinned Arena `.venv` carries `grpcio`/`protobuf` at runtime but intentionally not
`grpcio-tools`. Generate the stubs once into the git-ignored `_generated/` tree without
mutating the venv or `uv.lock`:

```bash
isaaclab_arena/integrations/cap_barrier/generate_perception_stubs.sh
```

The Kit + GPU smoke (not run in CI) streams to a running `cap_perception_bridge` node:

```bash
./dev_run.sh isaaclab_arena/scripts/run_cap_barrier_perception_stream.py \
    --headless --enable_cameras --device cuda:0
```

The stream runs for a bounded window (`--stream-seconds`, default 20 s) at
`--stream-hz` (default 10 Hz). Because the bridge fails closed on a stale or
absent producer, arm the ROS-side `cap_get_image_check` as an in-container poll
loop (~1 s cadence) **before** starting this producer rather than one docker-exec
attempt per check, or raise `--stream-seconds`.

This standalone script steps its own env locally and does NOT serve the CONTROL
barrier, so it is only for the Phase-3 `cap_get_image_check` path (bridge +
producer, no control plane). The GaP `get_observation` path bootstraps with
`ResetEpisode` and therefore needs a producer attached to the CONTROL barrier: use
the serve producer's `--perception-stream` flag so ONE Kit process both serves the
barrier and streams the camera (two simultaneous Kit processes is not supported):

```bash
isaaclab_arena/integrations/cap_barrier/generate_perception_stubs.sh
./dev_run.sh isaaclab_arena/scripts/run_cap_barrier_open_gripper_serve.py \
    --viz none --device cuda:0 --perception-stream 127.0.0.1:50061
```

`--perception-stream` implies `--enable_cameras` (the exterior_cam RTX spawn needs
it). Bringup ORDER matters: start this producer and wait for
`CAP_PRODUCTION_KIT_ENV_READY` BEFORE bringing up the CAP control plane; a control
plane started against no fresh producer will not go ACTIVE.

The serve loop samples the camera on the main/Kit thread right after each physics
step at ~10 Hz (decimated from the 200 Hz base) and offers frames to the same
nonblocking latest-frame producer; sampling never blocks or breaks the barrier
serve loop.

### Grocery-to-bin producer

The live GaP grocery walking skeleton uses a scene-specialized version of the same
external-policy serve loop:

```bash
isaaclab_arena/integrations/cap_barrier/generate_perception_stubs.sh
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
PYTHONPATH="$PWD" \
/home/rafael/Projects/Isaac-cap/external/IsaacLab-Arena/.venv/bin/python \
  -m isaaclab_arena.scripts.run_cap_barrier_grocery_to_bin \
  --viz none --device cuda:0 --perception-stream 127.0.0.1:50061
```

The fixed B=1 scene contains:

- `alphabet_soup_can_hope_robolab`, a dynamic graspable grocery;
- `grey_bin_robolab`, the calibrated packing destination;
- a local kinematic `procedural_table` collision support; and
- the RGB-D `exterior_cam`.

The object and bin poses are pinned from the successful DROID
`single_object_uv_rebase_seed71` rollout. The DROID base remains at world
identity, matching the `arena_droid_b1` base calibration used by the
camera-to-base, GraspGen, and MoveToPose frame chain. The producer emits
`CAP_GROCERY_TO_BIN_SCENE_READY` after the generation-1 bootstrap fence; the
marker names the exact object, bin, camera, and camera profile.

The default `--camera libero` uses the existing top-down CAP camera.
`--camera oblique` selects the existing Maple DROID agent-view camera as the
open-vocabulary fallback without changing the scene or base calibration. Both
publish the camera's live world pose; the ROS adapter derives `T_base_cam` from
that pose and the pinned identity `T_world_base`.

The live GPU acceptance is non-vacuous only when all of the following hold:

- Kit emits `CAP_PRODUCTION_KIT_ENV_READY`,
  `CAP_SERVE_KIT_GENERATION_1_ATTACHED`,
  `CAP_SERVE_KIT_BOOTSTRAP_FENCE_OK`, and the exact
  `CAP_GROCERY_TO_BIN_SCENE_READY` marker.
- Kit emits no `CAP_SERVE_KIT_PERCEPTION_SAMPLE_FAILED`; its terminal
  `CAP_SERVE_KIT_PERCEPTION_STREAM_TRACE` reports both `offered > 0` and
  `sent > 0`.
- The real DINO/SAM path returns a nonempty mask and OBB for the grocery object.
- The grocery graph reaches its successful terminal and all producer, control
  plane, and client processes shut down cleanly.

A scene-ready or `CAP_SERVE_KIT_DONE` marker alone is not a pass. Failure to
detect the grocery is terminal for the smoke and never fabricates an object pose.
