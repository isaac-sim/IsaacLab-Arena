# CAP Arena barrier smoke

This directory is the Kit-side half of the CAP ROS v0.1 shared-memory barrier. The
normative ABI remains `cap_backend_arena/protocol.hpp` in `isaac_ros_cap`; the checked-in
JSON and binary fixtures under `isaaclab_arena/tests/test_data/cap_barrier` are generated
by its test-only `cap_arena_abi_golden_dump` target.

The original runner remains a deliberately fixed integration smoke. It proves one B=1
Franka environment at a 200 Hz declared base:

1. bootstrap generation 1 with FENCE frames while Kit physics is frozen;
2. run a short joint-streaming trajectory in lockstep;
3. stage an Arena reset without a physics step and attach generation 2;
4. emit the reset fence, resume in hold, and shut down cleanly.

For each frame, Kit publishes joint state and waits for the matching controller command.
It keeps the shared phase at `COMMAND_READY` while applying a PHYSICS command and calling
`env.step()` exactly once. Only after the step returns does it release `AWAIT_STATE`.
FENCE commands are discarded without calling `env.step()`. This makes `AWAIT_STATE` a
real simulator-quiescence point for generation changes.

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

The final markers are `CAP_SMOKE_KIT_DONE` on the Kit side and
`CAP_SMOKE_ORCHESTRATOR_DONE` on the ROS side.

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
endpoints, then acquires `joint_streaming`, publishes through the validating relay, and invokes the
real reset action:

```bash
docker exec -it -u admin isaac_ros_dev_container bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   source /tmp/cap_binary_install/setup.bash && \
   source /tmp/cap_prod_clean_install/setup.bash && \
   /tmp/cap_prod_clean_build/cap_backend_arena/cap_production_smoke_client'
```

After the generation-1 bootstrap fence, Kit samples the owner generation and consumer-serviceability
latch before every next PHYSICS publication. It stops generation-1 publication as soon as the reset
becomes observable, waits for exact generation 2, resets Franka without advancing physics, attaches
only after the sidecar is serviceable, and emits the generation-2 fence. The
`CAP_PRODUCTION_KIT_GENERATION_2_DETECTED` marker includes the actual generation-1 physics count.
After the client and Kit markers appear, send Ctrl-C to terminal 1. Success requires
`CAP_PRODUCTION_CLIENT_RESET_ACTION_OK`, `CAP_PRODUCTION_KIT_DONE`, and a clean control-plane
shutdown.

## Deliberate boundaries

- The fixed controller timing roster is smoke data, not a CR-21 roster/discovery answer.
- `applied_torque` is carried as smoke-level effort data; real/sim World State effort
  parity remains open.
- Nonfinite commands and safety fallback composition remain CR-22 work.
- The production control plane owns reset admission and fence closure; the Kit producer remains
  ROS-free and observes only the CR-05 shared-memory contract.
