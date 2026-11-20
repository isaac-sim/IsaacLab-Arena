---
name: inspect-sim-visually
description: Gets images of the Isaac Sim viewer from an Isaac Lab-Arena simulation without desktop screenshots. Use when debugging an environment and you need to see the scene, for example placement after reset, an object falling through or flying off the table, a robot pose, or a view from another angle; also when --headless, a camera move, --record_viewport_video or env.render() gives no image. Do not use for recording evaluation videos or a policy's camera observations (run-experiment).
---

# Inspect the Simulation Visually

Ask the simulation for a PNG of its viewer and read the file. Never take a desktop or X11 screenshot without asking the
user: it captures their whole screen, and it needs a visible window.

| Need | Mode |
|---|---|
| Look around a running scene: other angles, pause, object positions, repeated looks | [Running sim](#running-sim) (default) |
| Images at fixed points of a script you are writing anyway | [Stills from a script](#stills-from-a-script) |
| Many images or a video as one picture | [Contact sheet](#contact-sheet) |

## Running sim

The simulation runs Isaac Sim's `isaacsim.code_editor.python_server`. From the host, `run_in_sim.py` sends it Python
code, and `sim_helpers.py` adds one-line helpers. One launch serves any number of looks.

1. **Launch** with a free port in 18200-18299 and outputs outside the repository. Take the runtime prefix from
   `arena-runtime`; the Docker form is:

   ```bash
   docker exec -e OPENBLAS_NUM_THREADS=1 -w /workspaces/isaaclab_arena --user "$(id --user --name)" <container> bash -c \
     "env -u DISPLAY setsid nohup isaaclab.sh -p isaaclab_arena/evaluation/policy_runner.py \
        --policy_type zero_action --num_steps 100000 --num_envs 1 --output_base_dir /tmp/<run>/outputs \
        --kit_args '--enable isaacsim.code_editor.python_server --/exts/isaacsim.code_editor.python_server/port=<port>' \
        --viz kit <environment> [environment args] > /tmp/<run>/run.log 2>&1 < /dev/null & echo \$!"
   ```

   It prints the run's session id; keep it for step 4.

   - `--viz kit` with `DISPLAY` unset starts the viewer without a window. The viewer then updates Kit every step,
     which the server needs to answer. Do not set `HEADLESS=1` here.
   - Without a policy, `isaaclab_arena/scripts/environment_runner.py --viz kit --device cpu --kit_args '...'
     <environment>` also works (PhysX, one env).
   - Add `--enable_cameras` when the run needs camera observations.
   - `/tmp` is shared with the container and the container uses host networking, so the host client and `Read` work
     on the same paths.

2. **Wait for the environment and load the helpers.** The server answers during app start, then pauses while the
   environment builds (40-50 s after launch when warm; the first launch in a new container can stay silent for about
   8 minutes). Until the environment exists, send only the helpers and `find_arena_env()`: Isaac Lab removes its
   modules from `sys.modules` while the app starts, and a request that imports an Isaac Lab or Arena module then
   breaks the run.

   ```bash
   C=skills/developer/inspect-sim-visually/scripts
   for attempt in $(seq 60); do
     python3 -I $C/run_in_sim.py --port <port> --file $C/sim_helpers.py --timeout 10 > /dev/null 2>&1 &&
       python3 -I $C/run_in_sim.py --port <port> --code 'find_arena_env().common_step_counter' --timeout 10 && break
     grep -q "Command failed" /tmp/<run>/run.log && { tail -30 /tmp/<run>/run.log; break; }
     sleep 5
   done
   ```

3. **Look.** Send one call per request and `Read` each PNG:

   ```bash
   python3 -I $C/run_in_sim.py --port <port> --code 'set_camera(eye=[0.5, -0.7, 0.2], target=[0.45, 0.0, 0.05], env_index=0)'
   python3 -I $C/run_in_sim.py --port <port> --code 'capture_viewer("/tmp/<run>/side.png")' --wait_for_file /tmp/<run>/side.png
   ```

   | Call | Effect |
   |---|---|
   | `capture_viewer(path, width=640, height=360)` | Writes the viewer image about 0.5 s later; pass `--wait_for_file path` |
   | `set_camera(eye, target, env_index=None)` | Points the viewer camera; meters, relative to env `env_index`'s origin, else world frame |
   | `print_scene_state(env_index=0)` | Position relative to the env origin and speed of every rigid object and articulation |
   | `pause_physics()` / `resume_physics()` | Freezes the simulation, including the episode clock, for several looks at one state (PhysX only) |
   | `pause_when(condition)` | Pauses on the first update where `condition(env)` is true, for events faster than a request, e.g. `pause_when(lambda env: env.arena_world.get_root_linear_velocity_w("dex_cube")[0, 2] < -1.0)` (PhysX only) |
   | `move_object(scene_key, position, env_index=0)` | Teleports a rigid object to `position` relative to the env origin, at rest |
   | `find_arena_env()` | The unwrapped Arena env, for anything else, e.g. `find_arena_env().scene["dex_cube"]` |

   Requests run inside Kit's update loop. Never call `app.update()`, `sim.step()`, `sim.render()` or
   `render_rgb_array()` in a request. Variables a request defines stay available to later requests. A pause takes
   effect after the env step in progress, so the frozen state is one step past the trigger.

4. **Stop** the run when done; the server runs any code sent to its port. Stop the whole session, because Kit's
   telemetry process outlives the main process:

   ```bash
   docker exec <container> bash -c 'pkill -TERM -s <sid>; sleep 5; pkill -KILL -s <sid>; pgrep -s <sid> || echo stopped'
   ```

## Stills from a script

For a script that builds the env itself (see `main()` in `isaaclab_arena/scripts/environment_runner.py` for the launch
pattern), launch with `HEADLESS=1`, `DISPLAY` unset, and
`--viz kit --experience isaaclab.python.headless.rendering.kit` (`--enable_cameras` also works):

```python
env_cfg, env_kwargs = arena_builder.build_env_cfg()
env_cfg.viewer.resolution = (640, 360)  # default 1280x720
env = arena_builder.make_registered(env_cfg, env_kwargs)
env.reset()
viewer = next(v for v in env.unwrapped.sim.visualizers if v.cfg.visualizer_type == "kit")

from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
from PIL import Image

set_kit_renderer_camera_view(eye=[0.5, -0.7, 0.2], target=[0.45, 0.0, 0.05], camera_prim_path="/OmniverseKit_Persp")
for _ in range(3):  # the first frame after a change is noisy
    frame = viewer.render_rgb_array()  # uint8, (height, width, 3)
Image.fromarray(frame).save("/tmp/<run>/side.png")
```

With `HEADLESS=1 --viz kit`, camera observations (`obs["camera_obs"]`) only update on steps after a
`render_rgb_array()` call. Do not use this mode for runs whose policy reads cameras.

## Contact sheet

Tile frames into one image of at most about 1568 px width:

```bash
ffmpeg -v error -y -pattern_type glob -i '/tmp/<run>/cap_*.png' -vf "scale=320:-2,tile=4x3:padding=4" -frames:v 1 /tmp/<run>/sheet.png
ffmpeg -v error -y -i clip.mp4 -vf "select='not(mod(n\,5))',scale=320:-2,tile=4x3:padding=4" -frames:v 1 /tmp/<run>/sheet.png
```

`tile=4x3` holds 12 panels and leaves the rest black, so match the grid to the number of images. The second command
takes every 5th frame of the video; change the 5 to spread the panels over the clip.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `OverrideParseException: --headless` | The flag no longer exists | Unset `DISPLAY` (running sim) or set `HEADLESS=1` (script) |
| `No reply ... within` during the rollout | Launched with `HEADLESS=1`, or without `--viz kit` and cameras: nothing updates Kit while stepping | Relaunch as in step 1. During app start or env build the pause is normal |
| Capture shows the previous view | Camera move and capture in one request | Send them as separate requests |
| Camera move has no effect in a headless script | `sim.set_camera_view()` and the viewer's `set_camera_view()` do not reach the headless capture | `set_kit_renderer_camera_view(...)` as above |
| Capture shows only the ground plane | Launched without `--viz kit`; scene partitioning hides the envs | Add `--/rtx/scenePartitioning/showAllPartitionsByDefault=true` to `--kit_args` |
| `NameError: name 'capture_viewer' is not defined` | New simulation process | Send `sim_helpers.py` again |
| `TypeError: SimulationCfg.physics must be a concrete PhysicsCfg` while the env builds | A request imported an Isaac Lab or Arena module while the app was starting | Relaunch; send only the helpers and `find_arena_env()` until the env exists |
| Red, green and blue arrows in the image | Isaac Lab debug markers for frames such as the goal and the end effector | Not scene objects |
| `AssertionError: The Kit timeline is not playing` | Newton run; pausing does not apply | Capture while it steps |
| `ModuleNotFoundError: No module named 'omni.replicator'` | Script launched without a rendering experience | Add `--experience isaaclab.python.headless.rendering.kit` or `--enable_cameras` |
| `--record_viewport_video` writes no video | Isaac Lab no longer renders `render_mode="rgb_array"`; `env.render()` returns `None` | Use this skill's captures |
| `/isaac-sim/skills/isaac-sim-remote` scripts fail (`isaacsim.core.experimental`) or freeze the sim | Written for the Isaac Sim app, not Isaac Lab | Use `sim_helpers.py` |
| `outputs/` appears in the repository | `policy_runner.py` default output directory | Pass `--output_base_dir /tmp/<run>/outputs` |
| Faint copy of an object at its old place | Renderer temporal accumulation after a jump | Capture again |

Verified on 2026-11-20 in the Docker route (Isaac Sim 6.2.0-rc.14, Isaac Lab `e3f7d9ff1`) with `lift_object` (PhysX;
`policy_runner` and `environment_runner`; 1 and 4 envs) and `yam_cable_routing` (Newton, with and without
`--enable_cameras`). The native uv route is not verified; check that the `isaacsim.code_editor.python_server`
extension loads there.
