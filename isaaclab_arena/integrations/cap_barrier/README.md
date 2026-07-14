# CAP Arena barrier smoke

This directory is the Kit-side half of the CAP ROS v0.1 shared-memory barrier. The
normative ABI remains `cap_backend_arena/protocol.hpp` in `isaac_ros_cap`; the checked-in
JSON and binary fixtures under `isaaclab_arena/tests/test_data/cap_barrier` are generated
by its test-only `cap_arena_abi_golden_dump` target.

The current runner is deliberately a fixed integration smoke, not a production episode
or reset interface. It proves one B=1 Franka environment at a 200 Hz declared base:

1. bootstrap generation 1 with FENCE frames while Kit physics is frozen;
2. run a short joint-streaming trajectory in lockstep;
3. stage an Arena reset without a physics step and attach generation 2;
4. emit the reset fence, resume in hold, and shut down cleanly.

For each frame, Kit publishes joint state and waits for the matching controller command.
It keeps the shared phase at `COMMAND_READY` while applying a PHYSICS command and calling
`env.step()` exactly once. Only after the step returns does it release `AWAIT_STATE`.
FENCE commands are discarded without calling `env.step()`. This makes `AWAIT_STATE` a
real simulator-quiescence point for generation changes.

The ABI's `wait_interrupted` field is a consumer-serviceability latch. Deactivation asserts it,
generation publication preserves it, and the sidecar clears it only after its tick thread is live.
The Kit producer attaches only when generation and `AWAIT_STATE` match and the latch is zero. Its
initial zero is intentional because a newly created bootstrap barrier has no retiring consumer.

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

## Deliberate boundaries

- The fixed controller timing roster is smoke data, not a CR-21 roster/discovery answer.
- `applied_torque` is carried as smoke-level effort data; real/sim World State effort
  parity remains open.
- Nonfinite commands and safety fallback composition remain CR-22 work.
- Production startup, reset admission, and fence-close handshakes remain CR-04 work.
