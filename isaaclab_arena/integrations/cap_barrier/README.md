# CAP Arena barrier smoke

This directory is the Kit-side half of the CAP ROS v0.1 shared-memory barrier. The
normative ABI remains `cap_backend_arena/protocol.hpp` in `isaac_ros_cap`; the checked-in
JSON and binary fixtures under `isaaclab_arena/tests/test_data/cap_barrier` are generated
by its test-only `cap_arena_abi_golden_dump` target.

The original runner remains a deliberately fixed integration smoke. It is designed to prove one
B=1 DROID environment at a 200 Hz declared base:

1. bootstrap generation 1 with FENCE frames while Kit physics is frozen;
2. while the arm remains in hold, observe the physical finger move open-to-close-to-open;
3. run a short joint-streaming trajectory in lockstep;
4. stage an Arena reset without a physics step and attach generation 2;
5. emit the reset fence, resume in hold, and shut down cleanly.

Historical Franka smoke results do not transfer to this embodiment. DROID parity must be re-earned
after the producer revision is pinned, including an observed finger transition so an already-open
initial state cannot make the gripper path pass vacuously.

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
physical arm reaction is accumulated across the complete open-close-open transition before the
provisional `1e-4 rad` diagnostic bound is evaluated. Earlier 2026-07-17 runs on the pinned
`arena_droid_b1` scene with the real `robotiq_gripper_controller` stopped once at `2.82e-5 rad`
(`1e-5` bound), then stopped five times at the bit-identical `1.003742218e-4 rad` crossing (`1e-4`
bound). Those fail-fast values are censored and are not calibration maxima; the five identical
crossings establish determinism only for this fixed scene, seed, and physics configuration. The
final config-scoped tolerance requires at least five full-transition maxima. A scene, embodiment,
or physics configuration change invalidates that calibration, and any later observation above half
the final tolerance requires recalibration. Command-frame echo alone cannot satisfy this proof.

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

## Run

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

The Kit side must emit `CAP_SMOKE_KIT_GRIPPER_TRANSITION_OK` before
`CAP_SMOKE_KIT_DONE`; the ROS side finishes with `CAP_SMOKE_ORCHESTRATOR_DONE`.

## Production control-plane smoke

The production runner leaves the reset instant to the ROS Session Manager action. The smoke uses
three processes: the custom in-process ROS composition container, the ROS-free Kit producer, and a
test-only typed ROS client. Build `isaac_ros_cap` with `BUILD_TESTING=ON`; the client is intentionally
not installed and must be run from the colcon build tree.

In terminal 1, start the control plane inside the Isaac ROS development container. Replace the
build and install bases below if colcon used different paths:

```bash
docker exec -it -u admin isaac_ros_dev_container bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /tmp/cap_binary_install/setup.bash && \
   source /tmp/cap_prod_clean_install/setup.bash && \
   ros2 launch cap_backend_arena arena_control_plane.launch.py'
```

Wait for `CAP_CONTROL_PLANE_READY_FOR_KIT`, then start Kit in terminal 2 on the host:

```bash
cd /home/rafael/Projects/arena-cap-barrier
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
PYTHONPATH="$PWD" \
/home/rafael/Projects/Isaac-cap/external/IsaacLab-Arena/.venv/bin/python \
  isaaclab_arena/scripts/run_cap_barrier_production_smoke.py --viz none --device cuda:0
```

In terminal 3, start the typed test client. It waits for Kit-driven bootstrap and the production
endpoints, drives a guarded close then open gripper operation, releases that lease, then acquires
`joint_streaming`, publishes through the validating relay, and invokes the real reset action:

```bash
docker exec -it -u admin isaac_ros_dev_container bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /tmp/cap_binary_install/setup.bash && \
   source /tmp/cap_prod_clean_install/setup.bash && \
   /tmp/cap_prod_clean_build/cap_backend_arena/cap_production_smoke_client'
```

After the generation-1 bootstrap fence, Kit samples the owner generation and consumer-serviceability
latch before every next PHYSICS publication. It stops generation-1 publication as soon as the reset
becomes observable, waits for exact generation 2, resets DROID without advancing physics, attaches
only after the sidecar is serviceable, and emits the generation-2 fence. The
`CAP_PRODUCTION_KIT_GENERATION_2_DETECTED` marker includes the actual generation-1 physics count.
After the client and Kit markers appear, send Ctrl-C to terminal 1. Success requires
`CAP_PRODUCTION_CLIENT_GRIPPER_TRANSITION_OK`, `CAP_PRODUCTION_KIT_GRIPPER_TRANSITION_OK`,
`CAP_PRODUCTION_CLIENT_RESET_ACTION_OK`, `CAP_PRODUCTION_KIT_DONE`, and a clean control-plane
shutdown.

## Deliberate boundaries

- The fixed controller timing roster is smoke data, not a CR-21 roster/discovery answer.
- `applied_torque` is carried as smoke-level effort data; real/sim World State effort
  parity remains open.
- Arm-command finite-value policy and safety fallback composition remain CR-22 work; the DROID
  gripper endpoint is already finite and discrete at this producer boundary.
- The production control plane owns reset admission and fence closure; the Kit producer remains
  ROS-free and observes only the CR-05 shared-memory contract.
