# Run an Arena experiment with Astra

Use this document as the instructions for the agent coordinating the experiment.
The user supplies a typed Experiment YAML path. Arena is installed, this branch is
checked out, and the checkout's Docker container is running. Start the Experiment
Runner, assign each episode to a fresh Astra controller, and report Arena's results.
The user does not need to launch the runner or attach controllers manually.

You need host terminal access, tools for viewing local images, and the ability to
spawn controllers **without inheriting this conversation**. When the agent tool
supports `fork_turns`, use `fork_turns="none"`. Use the same Astra model for every
controller. If fresh controller conversations are unavailable, report that missing
capability before starting; do not reuse a conversation across episodes.

## Prepare the requested experiment

Work from the Arena checkout containing this document. Discover the host repository,
host username, and running container:

```bash
repository_root=$(git rev-parse --show-toplevel)
host_username=$(id -un)
docker ps --filter "volume=$repository_root" --format '{{.Names}}'
```

Select the container mounting this repository at `/workspaces/isaaclab_arena` and
retain its name as `arena_container`. Inspect mounts if there are multiple matches.
Use the already-running runtime and its supported pinned dependencies. A running
container does not guarantee compatible sources: this checkout is mounted into it.
Before creating an invocation, compare the pinned and checked-out Isaac Lab revisions:

```bash
git ls-tree HEAD submodules/IsaacLab
git -C submodules/IsaacLab rev-parse HEAD
```

Unless the user explicitly selects another supported Isaac Lab source directory,
these commits must match. If they differ, report both revisions and stop before
launching. Do not change the submodule or hide duplicate CLI arguments with an
argparse workaround. A conflict for `--enable_cameras` can indicate this mismatch.

When the user provides an explicit runtime selection, verify imports resolve to
those supported sources and apply the same environment to every Arena Python
process. Record the selected source directory and revision. Do not automatically
discover or reuse historical validation snapshots as a fallback.

Check the runner's CLI in a separate container process, as the host user, using the
selected environment:

```bash
docker exec "$arena_container" su "$host_username" -s /bin/bash -c \
  'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py --help'
```

This check does not launch the simulator or an episode. If it fails, report the
error and stop; do not rebuild Docker or change dependencies automatically.

Resolve the YAML path supplied by the user. Read its Run settings to establish the
requested scope: Run names, episode limits, rebuilds, environment counts, cameras,
and referenced paths. The session policy supports one DROID environment with absolute
joint actions. Preserve task instructions, seeds, Run order, and rollout limits.
Do not silently reduce the experiment to one task or episode. For example, the full
`robolab_openpi_jobs_config.yaml` requests 38 Runs with ten episodes each. State the
scope before launching; the user's request to run the supplied file authorizes it.

Create a unique invocation directory on the mounted checkout:

```bash
mkdir -p "$repository_root/outputs/astra_experiments"
invocation_directory_host=$(mktemp -d "$repository_root/outputs/astra_experiments/run_XXXXXXXX")
invocation_directory_container="/workspaces/isaaclab_arena/outputs/astra_experiments/$(basename "$invocation_directory_host")"
experiment_output_host="$invocation_directory_host/experiment"
experiment_output_container="$invocation_directory_container/experiment"
```

Keep setup records, the runner log, and input copies in the invocation directory.
Leave its `experiment` child absent: `--experiment_output_directory` requires a
missing or empty directory. Never overwrite a prior invocation.

Make the input YAML available inside the container. For a file in the checkout,
translate its repository prefix to `/workspaces/isaaclab_arena`. For an external
file such as one in Downloads, copy it unchanged to
`$invocation_directory_host/experiment_input.yaml`, then use the corresponding
container path. Referenced paths must also be available in the runtime; copying
the top-level file does not mount external files. Resolve a missing required input
before launching. Keep the selected container path as `experiment_path_container`.

Record the repository revision, input source/copy paths and hashes, selected model,
container, command, and fresh-conversation mechanism in `SETUP.md` in the invocation
directory. Copy this document there. Shell variables may not survive between tool
calls; retain the resolved values and substitute them explicitly when needed.
Keep prompts and configuration fixed throughout the experiment.

## Start the Experiment Runner

Run this command as a long-running terminal process and retain its process handle.
The positional arguments preserve paths containing spaces or shell characters:

```bash
docker exec "$arena_container" su "$host_username" -s /bin/bash -c \
  'cd /workspaces/isaaclab_arena && exec /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py "$@"' \
  -- arena-experiment \
  --experiment_config "$experiment_path_container" \
  --policy_config isaaclab_arena_examples/robot_tool_control/policy_configs/astra_droid.yaml \
  --viz none --record_camera_video \
  --experiment_output_directory "$experiment_output_container" \
  > "$invocation_directory_host/runner.log" 2>&1
```

Use only user-requested experiment overrides in addition to this command. Do not
start an OpenPI server: `--policy_config` replaces that policy for every Run.
Do not launch the standalone pose-command server or invoke IK.

Monitor startup and keep the runner alive while controllers work. Its log prints
`Session policy directory: ...` for each policy instance. Translate container paths
to the host checkout. You can also discover manifests with the following pattern,
restricted to **this invocation's** experiment output directory:

```text
<experiment-output>/<run>/policy/rebuild<N>/<instance-id>/session.json
```

Never attach to a session found in an earlier output directory. Startup can take
time; an absence of requests alone is not failure. Check the runner process and
log. Do not restart a failed launch or replace an interrupted attempt automatically.
If the user subsequently requests a setup fix and a new launch, preserve the failed
invocation, repeat readiness checks with the selected runtime, and create a new
invocation directory. A startup failure before any episode is distinct from a
controller's task outcome; do not use this recovery to replace a failed episode.

## Assign one fresh controller per episode

Inspect a discovered session with the standard-library client on the host, from
the checkout root:

```bash
python3 -m isaaclab_arena_examples.robot_tool_control.session_client \
  inspect --session /absolute/host/path/to/session
```

When an active request appears, identify its episode by
`(policy_instance_id, env_id, episode_index)`. Maintain a record of identities
already assigned and their controller IDs. Create exactly one controller for each
new identity, with **no inherited conversation history**. The coordinator handles
processes and episode boundaries; the controller chooses all robot actions.

Pass only the checkout path, session path, expected episode identity, and this
instruction, substituting the actual values:

```text
Work from <host-checkout-path>. Read <host-session-path>/controller_prompt.md.
Control only policy_instance_id=<id>, env_id=<id>, episode_index=<index>.
Use only that fixed prompt and the assigned episode's requests and observations.
View the supplied images and submit joint-action chunks with session_client.
Stop at this episode's boundary or any exchange error. Do not launch or reset
the simulator, serve another episode, or read earlier attempts or source code.
```

Do not pass this coordinator conversation, the experiment YAML, setup records,
prior controller reports, images, coordinates, or actions to the controller.
Record controller IDs and their episode identities in the invocation directory
for provenance. The filesystem protocol cannot itself erase model history.

Keep the same controller for all requests within its episode. After a chunk, it
uses the new observations and may correct its next actions. Do not spawn one
controller per chunk or give advice based on another episode.

Monitor the controller and runner in short intervals. Each request has a default
600-second wall-clock timeout; physics waits during inference. A null
`active_request` normally means a chunk is executing, not that the episode ended.
An answered request can also remain visible briefly; neither condition should
create another controller or response.

To detect an assigned episode's boundary reliably, inspect its own persistent
manifest:

```text
<session>/episodes/env<env_id>_episode<episode_index>/episode.json
```

Its status changes from `active` to `ended` or `stopped`. The session's `last_event`
may already describe the next episode when you poll. Wait for the previous
controller to stop before assigning the next episode to a fresh controller.
If it lingers after its boundary, stop that controller without supplying new
episode data. A closed session finishes one policy instance or rebuild; continue
monitoring for other Runs until the Experiment Runner exits.

Never replace a controller for an already-assigned episode or retry a failed
attempt. If a controller stops before its episode boundary, report the interruption
and preserve the attempt; do not reset or resubmit its last action. Let the pending
request's timeout terminate the Run through its normal error path. Do not enable
`--continue_on_error` unless the user requested it.

## Finish and report the result

Keep every successful, unsuccessful, and interrupted episode. An `episode_ended`
event is a boundary, not a success verdict. After the runner exits, check:

- `arena_experiment_result.json` for Run status and episode outcomes;
- `index.html` for the evaluation report;
- each Run's `episode_results_rebuild<N>.jsonl` and requested camera videos;
- closed policy sessions and stopped controller agents.

On a runner error, aggregate result files may not exist. Report its exit status,
the relevant log message, and any retained session `error.json` and episode records.
Do not wait indefinitely for an aggregate report after the process has exited.

Return the host output/report/video paths, completed versus requested episode
counts, actual task outcomes, and any errors or missing prerequisites. State how
fresh controller conversations were created. Do not launch another evaluation
or modify the prompt in response to an unsuccessful task result.
