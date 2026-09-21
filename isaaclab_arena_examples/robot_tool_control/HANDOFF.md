# DROID command demo handoff

The example connects a model conversation to a persistent DROID simulation through a local command queue.
It accepts one flange pose and gripper target per decision, executes bounded motion with differential IK,
and returns camera images, calibration, measured robot state, and the command outcome. Physics pauses
between commands. See [POSE_COMMANDS.md](POSE_COMMANDS.md) for the interface and launch instructions.

For another installed checkout with its Docker container and an Astra session already running,
[CONTROLLER_PROMPT.md](CONTROLLER_PROMPT.md) provides portable instructions for starting the simulator,
running one episode from current observations, exporting videos, and shutting down. Its container name,
username, and session paths are discovered at runtime. The original local prompt below is retained as a record
of the historical evaluation; it is not required to run the example.

## Implemented changes

- `server.py` maintains the simulation and records commands, observations, trajectories, and episode results.
- `client.py` submits requests atomically and waits for their responses without automatic retries.
- `export_videos.py` adds a two-second terminal-image pause to presentation copies, with no text overlay.
- The DROID differential IK action targets the Robotiq gripper flange, matching the measured end-effector pose.
  The end-effector marker remains at the fingertip grasp point used by other Arena controllers.
  Observation documentation specifies world coordinates and xyzw quaternions.
- Camera recording discards interrupted episodes on explicit reset and uses Isaac Lab's final observations
  to include the terminal frame before automatic reset.

## Recorded result

On 2026-09-11, a new Astra conversation completed one episode with seed 43: seven control commands,
376 simulation steps, and 25.07 simulated seconds. No previous episode coordinates or conversation history
were supplied. The model derived targets from current images and calibration, checking feedback between commands.
All five pose moves converged; there was no regrasp or explicit reset. Arena confirmed placement in the bowl.

This run used a separate `codex exec` process with `gpt-6-astra`, memory disabled, and no resumed or forked
conversation. The controller and prompt stayed fixed during the episode. Input restrictions were instructions,
not a separate filesystem sandbox. Earlier sessions were development attempts and recording revisions; they
should not be presented as independent first attempts.

Local artifacts are under `outputs/robot_tool_control/session_06_fresh_context/` and are not committed:

- `RESULT.md`, `CONTROL_REPORT.md`, and `COORDINATE_DERIVATION.md` describe the completed run.
- `CONTROLLER_PROMPT.txt` and `protocol.json` preserve the initial prompt, setup, and source hashes.
- `commands.jsonl`, `episodes.jsonl`, and `controller_events.jsonl` preserve execution records.
- `verification.json` and `video_verification.json` record the checks.
- `presentation_videos/robot-cam-env0-external_camera_2_rgb-episode-0-with-final-frame.mp4` is the main presentation video.
- `videos/` contains the raw recordings; both directories also contain the other external and wrist camera views.

All six videos decoded successfully. Raw recordings contain 376 frames, including the terminal frame;
presentation copies contain 406 frames at 15 fps. The extra two seconds pause the saved image and do not
extend simulation. No simulator or model-controller process remained running after the evaluation.

## Validation and runtime notes

The recorder unit checks passed during development: 18 tests, excluding the real-encoder test. The completed
simulation and six actual encoded videos additionally exercised the controller, success termination, and recording.
This is one successful episode, not a benchmark success-rate estimate.

The original working checkout had a pre-existing Isaac Lab submodule mismatch. Its checked-out revision was
`af1bab4dc173ba69b08fab779c14ead61d13fd33`, while Arena pinned
`bb0c8e1b9af381bf13064ec3303e17db79e4b6ef`. The demo left that checkout unchanged and used an extracted copy of the
pinned revision in `outputs/robot_tool_control/isaaclab_pinned/`. The launch set `ISAACLAB_PATH` to that directory
and put its immediate `source/` package directories on `PYTHONPATH`. Those local dependency files are not committed.
A checkout with the supported pinned dependencies can use the launch command in `POSE_COMMANDS.md`.

Before another evaluation, start a new model conversation and session directory. Supply the interface and that
episode's observations only; do not include these historical reports or previously derived targets. Keep the
outcome of every attempt, including failures, and stop after its first episode terminates.
