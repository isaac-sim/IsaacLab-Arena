# Run one DROID pick-and-place episode

Use this document as the instructions for the current model conversation. Control the simulated DROID robot
to pick up the Rubik's cube and place it in the bowl. Start the command server, select actions from its observations,
and close the server after the episode. The user has already installed Arena and started the Docker container
for this checkout. Your session must have terminal access and a tool for viewing local images.

## Keep the attempt independent

Use a conversation that has not seen earlier demo observations, derived target coordinates, action sequences,
or result reports. If this conversation already contains those, ask the user to open a new conversation and
provide this document there; stop before starting the simulation. Instructions cannot erase existing context.
General knowledge of the robot and command interface is allowed.

Run one episode. Keep the prompt, controller, motion limits, and selected seed fixed throughout it.
Do not reset, restart after a failed attempt, change source code, or reuse targets from another episode.
You may correct a missed or incomplete move using new observations within this episode.

During control, read only this document, the command client's interface if needed, and this episode's data.
Do not read HANDOFF.md, earlier reports or recordings, scene configurations, object assets, or simulator internals.
Do not ask another model or person for target coordinates. Numerical geometry calculations on this episode's
images, camera calibration, and measured robot state are allowed.

## Find the runtime and create a session directory

Set the terminal working directory to the Arena checkout containing this document. Resolve that repository's
root and discover its running container on the host:

```bash
repository_root=$(git rev-parse --show-toplevel)
docker ps --filter "volume=$repository_root" --format '{{.Names}}'
```

Identify the container that mounts this checkout at /workspaces/isaaclab_arena. If there is more than one match,
inspect their mounts to select the correct container. If none matches, report the missing prerequisite and stop.
Do not rebuild an image, change submodules, or alter another running container.

Use the discovered name as arena_container. Resolve the host username with id -un and run container commands
as that user. Create a new, empty directory for this attempt:

```bash
mkdir -p "$repository_root/outputs/robot_tool_control"
session_dir_host=$(mktemp -d "$repository_root/outputs/robot_tool_control/session_XXXXXXXX")
session_dir_container="/workspaces/isaaclab_arena/outputs/robot_tool_control/$(basename "$session_dir_host")"
```

Shell variables and functions may not persist between terminal tool calls. Keep the resolved container name,
username, host repository path, and both session paths in your context and use them explicitly in later calls.

Save SETUP.md inside this new session directory with those values, the selected seed, and the input restrictions.
Copy this document into the session directory as the prompt record. Use seed 42 unless the user selected another
seed before the attempt. Do not search prior results to choose a seed.

## Start the command server

Run this as a long-running terminal tool process or in a dedicated terminal, retaining its process handle.
Substitute the selected seed if needed:

```bash
docker exec "$arena_container" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && exec /isaac-sim/python.sh -m isaaclab_arena_examples.robot_tool_control.server --session_dir '$session_dir_container' --seed 42" \
  > "$session_dir_host/server.log" 2>&1
```

Wait for ready.json in the new host session directory, checking that the server process is still running.
Read server.log only to diagnose startup or execution problems. If startup fails, report the error and stop;
do not modify the environment or try another robot episode. A supported Arena installation must already use
the compatible dependencies for this branch.

ready.json contains the initial observation and workspace bounds. Open the camera PNGs using your image viewer.
Replace the /workspaces/isaaclab_arena prefix in returned paths with the resolved host repository path.

## Observe and choose one command at a time

Invoke the client inside the same container as the host user. Define this helper in a persistent shell,
or expand it into each terminal tool call using the resolved values:

```bash
robot_command() {
  docker exec "$arena_container" su "$(id -un)" -s /bin/bash -c \
    'cd /workspaces/isaaclab_arena && exec /isaac-sim/python.sh -m isaaclab_arena_examples.robot_tool_control.client "$@"' \
    -- robot-command --session_dir "$session_dir_container" "$@"
}
robot_command observe
```

The literal robot-command supplies the shell's argument-zero value; subsequent client arguments are forwarded
as separate arguments, preserving quoted notes.

| Command | Arguments |
| --- | --- |
| observe | Save current observations without advancing physics. |
| move_to | --position X Y Z; optional --quaternion X Y Z W, --gripper 0 or 1, and --steps N. |
| set_gripper | --gripper 0 or 1; optional --steps N. Holds the measured flange pose. |
| wait | Optional --steps N. Holds the measured flange pose and last commanded gripper target. |
| shutdown | Close the server and finish cleanup after control stops. |

All commands accept --note with a short explanation of the current visual evidence and intended motion.
Never issue reset. Never queue several moves in advance. Select one command, wait for its response, inspect
the returned images and state, then decide what to do next.

Positions are meters in the robot-base frame. The controlled point is the Robotiq base_link gripper flange,
not the fingertip center. Quaternions use unit xyzw order. Gripper 0 opens and 1 closes. Omitting orientation
preserves the measured orientation; omitting gripper preserves its last commanded target.

The observations contain the task instruction, measured flange and finger-link poses, joint and gripper state,
robot world pose, and camera images and calibration. Calibration provides intrinsics, camera world positions,
and ROS optical-frame orientations in xyzw order. Convert world estimates into the robot-base frame using
the supplied robot pose. Object ground-truth poses are not exposed.

Derive object locations, grasp targets, and release targets from these observations. Account for the difference
between the flange and fingertips using the current images and measured robot geometry. Do not assume that a
closed gripper establishes a grasp or that opening over the bowl establishes placement.

The local controller advances its Cartesian reference by at most 0.004 meters and 0.04 radians per step,
then uses differential IK to track it. It does not avoid collisions. Workspace bounds are supplied in ready.json.
A move can finish early after at least six steps when position and rotation residuals are below 0.005 meters
and 0.05 radians. Steps is an execution budget from 1 through 240, not a list of model-generated actions.
Defaults are 180 for move_to and 24 for set_gripper or wait.

Responses include outcome, executed_steps, residual errors, episode_finished, and fresh observations.
ok=true means the request executed; it does not mean the motion converged or the task succeeded.
Use measured outcomes to correct subsequent commands within this episode. Physics pauses between requests.

If a client times out, its command may still execute. Inspect the response path printed by the client and
the server's status before taking another action. Never blindly resubmit or send another command while the
previous request's outcome is unknown.

## Finish and preserve the result

Continue until episode_finished is true, the user stops the run, or an execution error prevents further control.
The episode limit is 90 simulated seconds. Arena determines success, failure, or timeout; inspect the terminal
observation's termination terms. The simulator resets internally on termination, but the bridge preserves
the terminal observation and rejects further motion. Do not start another episode.

Write CONTROL_REPORT.md in this session directory. Include the observation sources, coordinate derivations,
commands and any corrections, terminal outcome, and simulation duration. Report failure or interruption
plainly. State whether earlier episode data was consulted. Keep all logs and observations.

Send shutdown once control has stopped and no motion request remains unresolved. Wait for the server process
to exit. Completed videos remain; shutting down an unfinished episode discards its partial videos, so preserve
the available observations and explain the interruption.

For a completed episode, export all three presentation videos:

```bash
docker exec "$arena_container" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m isaaclab_arena_examples.robot_tool_control.export_videos --session_dir '$session_dir_container'"
```

Presentation copies add a two-second pause on the terminal image, without a text overlay or extra simulation.
Confirm the exported files exist. Report the actual outcome and host paths to the report, raw videos, and
presentation videos. The server and model use instruction-based input restrictions, not an isolated filesystem.
