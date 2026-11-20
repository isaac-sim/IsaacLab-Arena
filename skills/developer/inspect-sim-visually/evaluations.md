# Inspect Sim Visually Evaluations

## Scenario 1: Scene A Few Steps After Reset

Setup: a ready Docker or native runtime; no simulation running.

Query: "Show me what `lift_object` looks like right after reset, then the gripper and cube from the side."

Expected behavior:

- Launches one `policy_runner.py` zero-action run (or `environment_runner.py`) with the Python server in `--kit_args`,
  `--viz kit`, `DISPLAY` unset, and `--output_base_dir` under `/tmp`.
- Waits with the skill's loop, sends `sim_helpers.py`, captures the default view, then moves the camera and captures
  again in a separate request; reads both PNGs.
- Stops the run with `pkill -f 'python_server/port=<port>'`.

Known failure modes:

- Takes a desktop or X11 screenshot.
- Passes `--headless`, or sets `HEADLESS=1` together with the Python server, so requests never return.
- Relaunches the simulation for every camera angle.
- Moves the camera with `sim.set_camera_view()` or adds a camera sensor to get a side view.

## Scenario 2: Object Falls Through The Table

Setup: a PhysX environment where an object drops through its support shortly after reset.

Query: "The cube falls through the table in my environment. Look at it from the side when it happens."

Expected behavior:

- Uses the running-sim mode and `pause_when` with a condition on the object's height or speed; polling with separate
  requests is too slow for a fall.
- Captures the frozen state from at least two camera poses with `set_camera(..., env_index=0)`, then
  `resume_physics()`.

Known failure modes:

- Calls `set_camera` and `capture_viewer` in the same request and reports the unchanged view as the new angle.
- Calls `app.update()`, `sim.render()` or `render_rgb_array()` inside a request.
- Uses `/isaac-sim/skills/isaac-sim-remote/scripts/viewport_screenshot.py`.

## Scenario 3: Stills From A Script With Several Envs

Setup: the agent is writing its own debug script that builds the environment with 4 envs.

Query: "In my debug script, save a picture of env 2 after reset."

Expected behavior:

- Launches the script with `HEADLESS=1`, `DISPLAY` unset, `--viz kit` and
  `--experience isaaclab.python.headless.rendering.kit` (or `--enable_cameras`).
- Moves the camera with `set_kit_renderer_camera_view` to env 2's origin plus an offset
  (`env.unwrapped.scene.env_origins[2]`), calls `render_rgb_array()` a few times, saves the last frame.

Known failure modes:

- Launches without a rendering experience and hits `ModuleNotFoundError: No module named 'omni.replicator'`.
- Uses `--record_viewport_video` or `env.render()`, which produce no image.

## Scenario 4: Camera-Policy Run That Also Needs Viewer Images

Setup: an environment whose policy reads `obs["camera_obs"]`.

Query: "While this camera policy runs, show me the viewer every few hundred steps."

Expected behavior:

- Keeps the viewer updating every step: `--viz kit` with `DISPLAY` unset plus `--enable_cameras`.
- Captures through the Python server.

Known failure modes:

- Uses `HEADLESS=1 --viz kit`, which freezes camera observations between viewer captures and stops the server from
  answering.
