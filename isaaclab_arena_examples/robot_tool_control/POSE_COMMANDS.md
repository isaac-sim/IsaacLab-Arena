# Control DROID through pose commands

This example keeps one DROID pick-and-place environment running and accepts commands through a local file queue.
The server has no model dependency. A person or a model session can inspect the saved camera images, choose a command,
and inspect the resulting robot state. Physics does not advance while the server waits for the next command.

## Run with an existing Astra session

Use the demo branch in an installed Arena checkout whose Docker container is already running. Open an Astra
conversation with terminal access to that checkout and a tool for viewing local images. Give it this instruction:

```text
Read isaaclab_arena_examples/robot_tool_control/CONTROLLER_PROMPT.md and run one DROID pick-and-place episode.
```

The [controller prompt](CONTROLLER_PROMPT.md) tells the session how to discover the container, create a new output
directory, start the simulator, inspect observations, issue commands, export videos, and shut down. It uses the
existing model session; no separate model service or API client setup is needed for this example.

The conversation must not contain earlier demo observations, target coordinates, action sequences, or result
reports. If it does, open a new conversation before giving the instruction. Each attempt uses its own observations
and runs one episode, including any corrections made from feedback within that episode.

Arena and the container must already use the dependencies supported by this branch. The workflow reports missing
prerequisites instead of changing the runtime. It does not rely on the original author's local outputs or paths.

For development history, recorded results, and validation, see [HANDOFF.md](HANDOFF.md). Do not supply that historical
report to a conversation that will control a new episode.

## Manual server and client use

Use a checkout with its supported Docker environment and pinned dependencies ready; see the
[dev-container skill](../../skills/developer/dev-container/SKILL.md). From the repository root on the host, discover
the container mounting this checkout and start the server as your host user:

```bash
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m isaaclab_arena_examples.robot_tool_control.server \
    --session_dir /workspaces/isaaclab_arena/outputs/robot_tool_demo'
```

Choose a new session directory for each server run. The default task places a Rubik's cube in a bowl, with seed 42
and a 90-second simulation limit. To use a banana, add `--object banana_ycb_robolab`; `--seed` selects the initial
placement. Wait for `Robot command server ready` before submitting commands.

In another host terminal, repeat the container discovery above and open a shell:

```bash
docker exec -it "$ARENA_CONTAINER" su "$(id -un)"
```

Inside that container shell, define a helper for this session:

```bash
robot_command() {
  /isaac-sim/python.sh /workspaces/isaaclab_arena/isaaclab_arena_examples/robot_tool_control/client.py \
    --session_dir /workspaces/isaaclab_arena/outputs/robot_tool_demo "$@"
}
robot_command observe
```

The response contains `observation.images`, mapping camera names to PNG paths, together with robot pose, joint and
gripper measurements, camera calibration, and the task instruction. To inspect an image from the host, replace its
`/workspaces/isaaclab_arena/` prefix with this checkout's path. A model session can open these files with its image
viewer, submit one command, and then open the images returned by that command. Object poses are not exposed.

| Command | Arguments and behavior |
| --- | --- |
| `observe` | Save current images and robot state without stepping physics. |
| `move_to` | Requires `--position X Y Z`. Optional `--quaternion X Y Z W`, `--gripper 0` or `1`, and `--steps`. |
| `set_gripper` | Requires `--gripper 0` to open or `1` to close. Holds the current flange pose for `--steps`. |
| `wait` | Hold the current flange pose and last commanded gripper state for `--steps`. |
| `reset` | Explicitly start another episode after inspecting the previous result. |
| `shutdown` | Close the environment and finish writing recordings. |

All commands accept `--note "description"` and `--timeout SECONDS` (default 180). Positions are meters in the robot-base
frame. The controlled tool frame is the Robotiq `base_link` gripper flange, **not the fingertip center**. Quaternions
use **xyzw** order and must have unit length. Omitting the quaternion preserves the measured orientation; omitting
the gripper target preserves its last commanded state. A requested gripper state applies throughout a move.

For a pose command, set `target_base_x`, `target_base_y`, and `target_base_z` from this episode's current images,
camera calibration, and measured pose:

```bash
robot_command move_to --position "$target_base_x" "$target_base_y" "$target_base_z" --note "Move to the observed target"
```

Each motion command returns new images and an `outcome`. A `move_to` reports `converged` after its reference reaches
the goal, measured position error is below 5 mm, and rotation error is below 0.05 rad, after at least six steps.
It reports `incomplete` when its step budget runs
out; use the measured pose and residual errors to choose the next command. `ok: true` means the command executed,
not that the motion converged or the task succeeded. Opening or closing the gripper does not establish a grasp.

The server bounds flange targets to `[0.05, -0.65, 0.10]` through `[0.85, 0.65, 0.80]` meters. Its reference pose advances
by at most 4 mm and 0.04 rad per simulation step. The controller uses the full error from the measured pose to that
reference, so corrective commands can grow when the arm falls behind. These limits do not provide collision avoidance.
The default budget is 180 steps for `move_to` and 24 for `set_gripper` or `wait`; `--steps` must be between 1 and 240.

Arena's task termination terms determine success. The episode ends on the first simulation step satisfying a
success, failure, or timeout condition. The response reports `episode_finished` and preserves its terminal
observation and termination terms before Isaac Lab resets internally. Further motion is rejected until an explicit
`reset`. Inspect the final images and result before starting another episode.

For evaluation, use a fresh model conversation for each attempt and evaluate one episode. Provide the task
instruction, tool interface, and observations from that episode only. Do not provide or read earlier episodes'
coordinates, action sequences, reports, or conversational history. Within the episode, use new camera observations
and measured command outcomes to choose and correct subsequent actions.

Fix the prompt, controller, command limits, and evaluation setup before each attempt; do not change them during
the episode. Stop at success, failure, or timeout. Do not reset or retry to replace a failed attempt. Keep every
attempt's session directory and record its outcome, including failures and interruptions. Partial videos are
discarded on an unfinished episode's reset or shutdown, as described below.

This protocol relies on instructions to keep earlier episodes out of the model's context. The example does not
provide a filesystem sandbox that prevents access to historical artifacts.

The client publishes `requests/<id>.json` atomically and waits for `responses/<id>.json`. Rejected commands return
`ok: false` and exit status 1. A client timeout does not cancel the request: it may still execute. Check the response
path printed by the client before submitting another command; the client never retries automatically.

The session directory retains `commands.jsonl` with requests and responses, `trajectory.jsonl` with measured motion
and residual errors, `episodes.jsonl` with Arena episode results, and camera snapshots under `observations/`.
Terminal snapshots are saved as `episode_<index>_terminal/` and referenced by `latest_terminal.json`. The three robot
camera videos are written under `videos/` for completed episodes; viewport video is not recorded. An explicit reset
or shutdown during an unfinished episode closes its encoders and discards its partial videos. Completed videos
remain after shutdown. Video follows simulation steps, so time spent deciding between commands is not included.

The example enables Isaac Lab's final observations so camera videos include the success or failure frame before
automatic reset. To give viewers time to inspect that frame, export presentation copies after the episode ends:

```bash
/isaac-sim/python.sh isaaclab_arena_examples/robot_tool_control/export_videos.py \
  --session_dir outputs/robot_tool_demo
```

This writes `presentation_videos/*-with-final-frame.mp4`, appending the saved terminal image for two seconds.
It also works for older recordings that omitted the terminal frame. The pause changes playback duration only;
episode termination and simulation time remain as recorded. Use `--episode` to select an episode and
`--hold_seconds` to change the display duration.
