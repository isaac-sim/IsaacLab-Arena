---
name: inspect-sim-visually
description: Gets images of an Isaac Lab-Arena simulation without desktop screenshots, from the Kit viewer at any angle and from the policy's cameras. Use when you need to visually inspect a scene or a running simulation, or when --record_viewport_video or env.render() gives no image. Do not use for recording evaluation videos (run-experiment).
---

# Inspect the Simulation Visually

Ask the running simulation for PNGs and read the files. Never take a desktop or X11 screenshot without asking the
user: it captures their whole screen, and it needs a visible window.

The simulation runs Isaac Sim's `isaacsim.code_editor.python_server`. From the host, `run_in_sim.py` sends it Python
code, and `sim_helpers.py` adds one-line helpers. One launch serves any number of looks.

1. **Launch.** The server runs any Python sent to its port, without authentication, and listens on the host's
   loopback (the container uses host networking). While the run lives, any user or process on this machine can run
   code as you through it. On a host shared with other users, ask the user before launching.

   Pick a port in 18200-18299 that nothing listens on: `ss -Hltn 'sport = :<port>'` must print nothing. The server
   binds with `SO_REUSEPORT`, so a second run on a busy port starts without an error and receives some of the
   requests. Take the runtime prefix from `arena-runtime`; the Docker form is:

   ```bash
   mkdir -p /tmp/<run>
   docker exec -e OPENBLAS_NUM_THREADS=1 -w /workspaces/isaaclab_arena --user "$(id --user --name)" <container> bash -c \
     "env -u DISPLAY setsid nohup isaaclab.sh -p isaaclab_arena/evaluation/policy_runner.py \
        --policy_type zero_action --num_steps 20000 --num_envs 1 --output_base_dir /tmp/<run>/outputs \
        --kit_args '--enable isaacsim.code_editor.python_server --/exts/isaacsim.code_editor.python_server/port=<port>' \
        --viz kit <environment> [environment args] > /tmp/<run>/run.log 2>&1 < /dev/null & echo \$!"
   ```

   It prints the run's session id; keep it for step 5. `--num_steps 20000` ends a forgotten run after roughly
   20-40 minutes; launch again if you need longer.

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

   From the repository root:

   ```bash
   C=skills/developer/inspect-sim-visually/scripts
   for attempt in $(seq 120); do
     python3 -I $C/run_in_sim.py --port <port> --file $C/sim_helpers.py --timeout 10 > /dev/null 2>&1 &&
       python3 -I $C/run_in_sim.py --port <port> --code 'find_arena_env().common_step_counter' --timeout 10 2> /dev/null && break
     grep -q "Command failed" /tmp/<run>/run.log && { tr '\r' '\n' < /tmp/<run>/run.log | tail -30; break; }
     sleep 5
   done
   ```

   The loop prints the current step once the environment is ready. If it ends without one, run the last call by hand
   to see the error.

3. **Look.** Send one call per request and `Read` each PNG:

   ```bash
   python3 -I skills/developer/inspect-sim-visually/scripts/run_in_sim.py --port <port> \
     --code 'set_camera(eye=[0.5, -0.7, 0.2], target=[0.45, 0.0, 0.05], env_index=0)'
   python3 -I skills/developer/inspect-sim-visually/scripts/run_in_sim.py --port <port> \
     --code 'capture_viewer("/tmp/<run>/side.png")' --wait_for_file /tmp/<run>/side.png
   ```

   The first capture after a camera move can still be blurry or show the old view. Wait about a second after
   `set_camera`, or capture twice.

   | Call | Effect |
   |---|---|
   | `capture_viewer(path, width=640, height=360)` | Writes the viewer image about 0.5 s later; pass `--wait_for_file path` |
   | `set_camera(eye, target, env_index=None)` | Points the viewer camera; meters, relative to env `env_index`'s origin, else world frame. Keep the eye offset sideways from the target: straight down gives an arbitrary roll |
   | `print_scene_state(env_index=0)` | Position relative to the env origin, speed, and tilt of every rigid object and articulation |
   | `pause_physics()` / `resume_physics()` | Freezes the simulation, including the episode clock, for several looks at one state |
   | `pause_when(condition)` | Pauses on the first update where `condition(env)` is true, for events faster than a request; can be armed while paused. A new call replaces the previous condition; join several with `or` |
   | `is_paused()` | Whether the simulation is paused; the reply is `true` or `false`. Wait for a `pause_when` condition with `until [ "$(python3 -I <scripts>/run_in_sim.py --port <port> --code 'is_paused()')" = true ]; do sleep 0.5; done` |
   | `save_policy_camera_images(output_dir, env_index=0)` | Saves the camera images the policy got at the last step (`obs_buf["camera_obs"]`), one PNG per camera term; needs `--enable_cameras` |
   | `end_episode()` | Starts a new episode in every env at the next step, through Isaac Lab's reset request |
   | `move_object(scene_key, position, env_index=0)` | Teleports a rigid object to `position` relative to the env origin, at rest |
   | `find_arena_env()` | The unwrapped Arena env, for anything else, e.g. `find_arena_env().scene["robot"].data.joint_pos.torch` (Isaac Lab data fields need `.torch`) |

   Requests run inside Kit's update loop. Never call `app.update()`, `sim.step()`, `sim.render()` or
   `render_rgb_array()` in a request. Variables a request defines stay available to later requests. The reply shows
   a value only when the code is a single expression; use `print()` in longer code.

   To see the scene as assembled, catch the start of a new episode instead of waiting for the current one to end:

   ```bash
   python3 -I skills/developer/inspect-sim-visually/scripts/run_in_sim.py --port <port> \
     --code 'pause_when(lambda env: int(env.episode_length_buf[0]) == 2); end_episode()'
   ```

   The run pauses within about a second. The new episode draws its own layout, so it differs from the one before. With
   absolute joint-position robots such as DROID, the zero-action policy is already moving the arm toward zero joint
   angles at that point.

   When checking how a scene assembles, also read the placement lines in the run log:
   `grep -E "\[placement\]|best-loss" /tmp/<run>/run.log`. A check that passes 0 of the candidate layouts, or
   `Falling back to best-loss layouts`, means the scene runs with a layout that breaks its relations.

   Pausing needs a playing Kit timeline: PhysX runs, or Newton runs with `--viz kit`. A pause takes effect a step or
   two after the trigger. For a moment inside an event, combine conditions, such as an object below a height and
   moving down. Episodes reset when they time out or succeed, which undoes a staged event; check
   `find_arena_env().max_episode_length - find_arena_env().episode_length_buf[0]` first.

   The default viewer camera can sit outside large scenes such as kitchens. Aim with `set_camera` at positions from
   `print_scene_state`. For surfaces and other background geometry, USD bounding boxes are fine, but those of moving
   objects, and of a robot that placement moves at reset, show their spawn pose while Fabric is on.

4. **Combine images** (optional). To see many captures at once, tile them into one labeled picture of at most about
   1568 px width. First bring every image to the same size and pixel format and label it with its file name (names
   without `:`); ffmpeg's tile filter silently drops panels when sizes or formats differ, as between viewer captures
   (RGBA) and camera images (RGB):

   ```bash
   for f in /tmp/<run>/cap_*.png; do
     ffmpeg -v error -y -i "$f" -vf "scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2,format=rgb24,drawtext=text='$(basename "$f" .png)':x=6:y=6:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6" "/tmp/<run>/label_$(basename "$f")"
   done
   ffmpeg -v error -y -pattern_type glob -i '/tmp/<run>/label_cap_*.png' -vf "tile=3x2:padding=4" -frames:v 1 /tmp/<run>/sheet.png
   ```

   A grid holds columns x rows panels and leaves the rest black, so match it to the number of images.

5. **Stop** the run when done; the server runs any code sent to its port. Stop the whole session, because Kit's
   telemetry process outlives the main process:

   ```bash
   docker exec <container> bash -c 'pkill -TERM -s <sid>; sleep 5; pkill -KILL -s <sid>; pgrep -s <sid> || echo stopped'
   ```

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `OverrideParseException: --headless` | The flag no longer exists | Launch with `--viz kit` and `DISPLAY` unset as in step 1 |
| `No reply ... within` during the rollout | Launched with `HEADLESS=1`, or without `--viz kit` and cameras: nothing updates Kit while stepping | Relaunch as in step 1. During app start or env build the pause is normal |
| Capture shows the previous view | Camera move and capture in one request, or captured right after the move | Send them as separate requests and wait about a second |
| Capture is completely black | The camera eye is inside geometry, such as a cabinet | Move the eye |
| Capture shows only the ground plane | Launched without `--viz kit`; scene partitioning hides the envs | Add `--/rtx/scenePartitioning/showAllPartitionsByDefault=true` to `--kit_args` |
| `NameError: name 'capture_viewer' is not defined` | New simulation process, or another run shares the port | Send `sim_helpers.py` again; if it keeps returning, check the port with `ss` and stop the other run |
| `TypeError: SimulationCfg.physics must be a concrete PhysicsCfg` while the env builds | A request imported an Isaac Lab or Arena module while the app was starting | Relaunch; send only the helpers and `find_arena_env()` until the env exists |
| Red, green and blue arrows in the image | Isaac Lab debug markers for frames such as the goal and the end effector | Hide with `find_arena_env().command_manager.set_debug_vis(False)` and `set_debug_vis(False)` on frame sensors, e.g. `find_arena_env().scene["ee_frame"]` |
| `AssertionError: The Kit timeline is stopped` | Newton run launched without `--viz kit` | Capture while it steps, or relaunch with `--viz kit` |
| Run ends with `step_index must advance by exactly one per environment between resets` | Seen once in about a dozen pauses in Arena's progress tracker; not reproduced | Relaunch |
| `--record_viewport_video` writes no video | Isaac Lab no longer renders `render_mode="rgb_array"`; `env.render()` returns `None` | Use this skill's captures |
| `/isaac-sim/skills/isaac-sim-remote` scripts fail (`isaacsim.core.experimental`) or freeze the sim | Written for the Isaac Sim app, not Isaac Lab | Use `sim_helpers.py` |
| `outputs/` appears in the repository | `policy_runner.py` default output directory | Pass `--output_base_dir /tmp/<run>/outputs` |
| Faint copy of an object at its old place | Renderer temporal accumulation after a jump | Capture again |

Verified from 2026-11-20 to 2026-11-25 in the Docker route (Isaac Sim 6.2.0-rc.14, Isaac Lab `e3f7d9ff1`) with
`lift_object` (PhysX; `policy_runner` and `environment_runner`; 1 and 4 envs), the 17 KitchenBench task configs
(`--env_spec`, DROID) and `yam_cable_routing` (Newton, with and without `--enable_cameras`). The native uv route is
not verified; check that the `isaacsim.code_editor.python_server` extension loads there.
