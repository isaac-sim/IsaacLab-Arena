# Inspect Sim Visually Evaluations

## Scenario 1: Scene A Few Steps After Reset

Setup: a ready Docker or native runtime; no simulation running.

Query: "Show me what `lift_object` looks like right after reset, then the gripper and cube from the side."

Expected behavior:

- Writes a one-Run zero-action Experiment and launches it with `experiment_runner.py`, the Python server in
  `--kit_args`, `--viz kit`, `DISPLAY` unset, and `--output_base_dir` under `/tmp`.
- Loads `sim_helpers.py`, arms `pause_at_next_episode()` before the environment is ready, captures the default view
  at the pause, then moves the camera and captures again in a separate request; reads both PNGs.
- Stops the run by the session id printed at launch (`pkill -s <sid>`) and confirms with `pgrep -s <sid>` that
  nothing is left.

Known failure modes:

- Takes a desktop or X11 screenshot.
- Stops only the main process, for example with `pkill -f 'python_server/port=<port>'`, and leaves Kit's
  telemetry process running.
- Passes `--headless`, or sets `HEADLESS=1` together with the Python server, so requests never return.
- Relaunches the simulation for every camera angle.
- Waits minutes for the current episode to time out instead of using `pause_at_next_episode()`.
- Moves the camera with `sim.set_camera_view()` or adds a camera sensor to get a side view.

## Scenario 2: Object Falls Through The Table

Setup: a PhysX environment where an object drops through its support shortly after reset.

Query: "The cube falls through the table in my environment. Look at it from the side when it happens."

Expected behavior:

- Uses `pause_when` with a condition on the object's height or speed; polling with separate
  requests is too slow for a fall.
- Captures the frozen state from at least two camera poses with `set_camera(..., env_index=0)`, then
  `resume_physics()`.

Known failure modes:

- Calls `set_camera` and `capture_viewer` in the same request and reports the unchanged view as the new angle.
- Calls `app.update()`, `sim.render()` or `render_rgb_array()` inside a request.
- Uses `/isaac-sim/skills/isaac-sim-remote/scripts/viewport_screenshot.py`.

## Scenario 3: One Env Among Several

Setup: an environment run with 4 envs.

Query: "Show me env 2 right after reset."

Expected behavior:

- Launches with `environment_builder: {num_envs: 4}`, pauses at the start of an episode with
  `pause_at_next_episode()`, and aims the camera at env 2 with `set_camera(..., env_index=2)` before capturing.

Known failure modes:

- Captures the default view and reports it as env 2; the default view follows env 0.
- Uses `--record_viewport_video` or `env.render()`, which produce no image.

## Scenario 4: Existing Experiment With A Camera Policy

Setup: an Experiment YAML whose policy reads `obs["camera_obs"]`, with its policy server running.

Query: "While this evaluation runs, show me the viewer and what the policy sees every few hundred steps."

Expected behavior:

- Launches the existing Experiment as it is with `experiment_runner.py`, adding the Python server in `--kit_args`,
  `--viz kit` with `DISPLAY` unset, and `--output_base_dir` under `/tmp`; does not write a new Experiment.
- Captures the viewer and calls `save_policy_camera_images` through the Python server.

Known failure modes:

- Writes a zero-action Experiment instead of inspecting the policy's own Run.
- Uses `HEADLESS=1 --viz kit`, which freezes camera observations between viewer captures and stops the server from
  answering.
