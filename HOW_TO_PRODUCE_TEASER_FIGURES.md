# How to produce the teaser figures

Commands used to generate the images and animations on the documentation landing page
(`docs/index.rst`). Grouped by the section of that page each asset belongs to.

All assets live under `docs/images/teaser_page/<section_folder>/`.

## Prerequisites

### Local environment-variable hacks

Several of these recordings depend on **uncommitted local hacks** in
`isaaclab_arena/environments/arena_env_builder.py`. Without them the commands below still run, but
you get the task's own camera and the robot in shot. Re-add them (or promote them to real CLI
flags) before reproducing these figures.

| Variable | Effect |
| --- | --- |
| `ARENA_VIEWER_EYE` | `x,y,z` camera position; setting it enables the whole override |
| `ARENA_VIEWER_LOOKAT` | `x,y,z` camera target (default `0,0,0`) |
| `ARENA_VIEWER_RESOLUTION` | `width,height` of the recorded mp4 (default `1920,1080`) |
| `ARENA_VIEWER_ORIGIN_TYPE` | `world` (default), `env`, `asset_root`, `asset_body` |
| `ARENA_NO_ROBOT` | `1` builds the scene without the embodiment |
| `ARENA_RESET_STEPS` | Force a time-out reset every N environment steps |

The viewer override also has to be re-applied to the video recorder: with a Kit visualizer active,
Isaac Lab overwrites the recording camera with the visualizer's own pose
(`VideoRecorder._sync_camera_from_visualizer`), which yanks the shared `/OmniverseKit_Persp`
viewport on the first captured frame.

`ARENA_NO_ROBOT` requires `--policy_type zero_action` — with no embodiment the action config is
empty, so a real policy fails on the action shape.

### Running in the container

Recordings run inside the clone's Docker container (see the `dev-container` skill), where the repo
is mounted at `/workspaces/isaaclab_arena`:

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c "cd /workspaces/isaaclab_arena && <command>"
```

Use the explicit interpreter path `/isaac-sim/python.sh` rather than `python`.

### Camera poses

Two poses are used throughout:

```bash
# Wide — frames a 2x2 grid of environments
ARENA_VIEWER_EYE=2.7621417857479345,-2.7132198781211803,2.5377173854675505
ARENA_VIEWER_LOOKAT=0,0,-0.55

# Close-up — single environment, fills the frame with one table
ARENA_VIEWER_EYE=1.859810866413229,-0.030319035672402704,0.7177487664395753
ARENA_VIEWER_LOOKAT=0,0,-0.2
```

To find a new pose, fly the viewport where you want it in the Kit GUI, select
`/OmniverseKit_Persp` in the Stage panel, and read `xformOp:translate` off the Property panel.

### Standard mp4 -> gif conversion

Used for every animated asset. 640 px wide, 12 fps, per-clip palette:

```bash
ffmpeg -i in.mp4 -filter_complex \
  "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  -loop 0 out.gif
```

Add `-t <seconds>` before `-i` to trim. A 100 s clip becomes a ~35 MB gif at these settings, so
trim before converting.

### Verifying recordings — important

**The renderer intermittently drops the scene mid-recording.** The run exits 0 with no error, but
every frame from some point onward is the flat empty backdrop. It hit 4 of 9 clips in one batch,
usually at a reset boundary. Re-running the same command fixes it.

Always check before publishing a clip. A blank frame has near-zero standard deviation:

```bash
for n in $(seq 10 20 590); do
  ffmpeg -v error -i clip.mp4 -vf "select=eq(n\,$n)" -vframes 1 -y /tmp/_chk.png
  convert /tmp/_chk.png -format "$n %[fx:standard_deviation]\n" info:
done
```

Values around `0.1`-`0.3` are real content; anything below `0.01` is blank. Sample densely —
checking only two frames misses clips that fail late.

## Swappable assets

`docs/images/teaser_page/object_swapping/` — one clip per swapped object, all in the same
environment.

Source mp4s were produced outside this session (one folder per task, each containing
`rl-video-step-0.mp4`), renamed to the object name and converted with the standard recipe at full
length:

```bash
for d in "$SRC"/droid_pnp_srl_openpi_*/; do
  name=$(basename "$d"); short=${name#droid_pnp_srl_openpi_}
  cp "$d/rl-video-step-0.mp4" "$DEST/$short.mp4"
done
```

These are still full length (12-60 s each, ~77 MB of gifs total). Trimming them to ~15 s the way
the Parallel Evaluation clips were trimmed would cut that to roughly 25 MB.

## Automatic Object Placement

`docs/images/teaser_page/automatic_object_placement/` — single environment, no robot, resetting
every 60 steps so each reset draws a fresh layout from the placement pool. 600 steps gives 10
layouts in a 12 s clip.

```bash
ARENA_VIEWER_EYE=1.859810866413229,-0.030319035672402704,0.7177487664395753 \
ARENA_VIEWER_LOOKAT=0,0,-0.2 \
ARENA_NO_ROBOT=1 \
ARENA_RESET_STEPS=60 \
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --enable_cameras --num_envs 1 \
  --policy_type zero_action --num_steps 600 \
  --record_viewport_video \
  --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/<task>.yaml \
  --output_base_dir outputs/teaser/placement/<task>
```

Then the standard gif conversion, full length.

Notes:

- No HDR override, so the backdrop stays the flat default grey.
- Re-placement on reset is controlled by `resolve_on_reset`, which defaults to `True`. The layout
  pool refills lazily, so layouts do not repeat however many resets you run.
- Layouts are deterministic for a given `--seed`, so re-running produces the same sequence. Pass a
  different `--seed` or `--placement_seed` for fresh variety.
- 60 steps (1.2 s) per layout is brisk — objects are still settling for the first third of each
  episode. Use `ARENA_RESET_STEPS=100` if you want each layout to read more clearly.
- **This is the configuration most affected by the blank-frame bug.** Verify every clip.

## Parallel Evaluation

`docs/images/teaser_page/parallel_evaluation/` — 4 environments per clip, a different HDR
background per task, driven by a real policy.

Requires the openpi policy server to be running.

```bash
ARENA_VIEWER_EYE=2.7621417857479345,-2.7132198781211803,2.5377173854675505 \
ARENA_VIEWER_LOOKAT=0,0,-0.55 \
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --enable_cameras \
  --num_envs 4 --env_spacing 1.5 --num_steps 5000 \
  --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
  --record_viewport_video \
  --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/<task>.yaml \
  light.hdr_image.hdr_names=["<hdr_name>"] light.hdr_image.enabled=True \
  --output_base_dir outputs/teaser/<task>
```

HDR names come from `isaaclab_arena/assets/hdr_image_library.py`. The mapping used:

| Task | HDR |
| --- | --- |
| `big_pumpkin_in_bin` | `home_office_robolab` |
| `bagels_on_plate` | `empty_warehouse_robolab` |
| `canned_food_in_bin` | `aerodynamics_workshop_robolab` |
| `mouse_on_keyboard` | `billiard_hall_robolab` |
| `rubiks_cube_and_banana` | `brown_photostudio_robolab` |
| `bbq_sauce_in_bin` | `blinds_robolab` |
| `small_pumpkin_in_bin` | `kiara_interior_robolab` |
| `mustard_in_left_bin` | `garage_robolab` |
| `clutter_pumpkin` | `wooden_lounge_robolab` |

Gifs are trimmed to the first 15 s (`-t 15` before `-i`), which keeps each around 5 MB instead of 35 MB.

`--num_steps 5000` gives a 100 s clip at 50 fps; each run takes roughly 18-20 minutes.

## Built-in Evaluation Environments

`docs/images/teaser_page/built_in_evaluation_environments/` — one still per environment, robot
removed, background keyed out to transparency.

Record a 1 s clip (`--num_steps 50` at 50 fps), which also makes the process exit on its own:

```bash
ARENA_VIEWER_EYE=1.859810866413229,-0.030319035672402704,0.7177487664395753 \
ARENA_VIEWER_LOOKAT=0,0,-0.2 \
ARENA_NO_ROBOT=1 \
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --enable_cameras --num_envs 1 \
  --policy_type zero_action --num_steps 50 \
  --record_viewport_video \
  --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/<task>.yaml \
  --output_base_dir outputs/teaser/no_robot/<task>
```

Extract the middle frame and key out the backdrop:

```bash
ffmpeg -i rl-video-step-0.mp4 -vf "select=eq(n\,25)" -vframes 1 frame.png

ARGS=""
for x in $(seq 0 40 1919); do ARGS="$ARGS -floodfill +$x+0 #E6E6E6 -floodfill +$x+1079 #E6E6E6"; done
for y in $(seq 0 40 1079); do ARGS="$ARGS -floodfill +0+$y #E6E6E6 -floodfill +1919+$y #E6E6E6"; done
convert frame.png -alpha set -fuzz 8% -fill none $ARGS cutout.png
```

Notes:

- No HDR override — the flat `#E6E6E6` backdrop is what makes the key clean.
- Flood-fill rather than `convert -transparent`: the backdrop grey is close to the light grey table
  legs and metal-bin highlights, so a global colour match punches holes in the furniture.
- Seeds ring all four edges every 40 px because objects touching an edge (the table legs) partition
  the backdrop into pockets that corner-only seeds cannot reach.
- Sanity check with the opaque-pixel fraction; these sit at 26-28%:
  `convert cutout.png -alpha extract -format "%[fx:mean]" info:`
- Composite over a contrasting colour to inspect:
  `convert -size 1920x1080 xc:magenta cutout.png -composite check.png`
- Edges are hard-cut, so anti-aliased boundary pixels keep a faint grey fringe. Visible if you
  composite onto a dark background.

## Environmental Variations

`docs/images/teaser_page/variations/` — source mp4s were produced outside this session.

Each gif is resampled to exactly 64 frames spread across the *whole* source clip, then played at
12 fps, so all four are 5.33 s and loop in sync. Resampling rather than truncating matters: the
variation sweep runs the full length of each clip, so cutting at frame 64 would discard half of it.

```bash
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 in.mp4)
ratio=$(awk -v d="$dur" 'BEGIN{printf "%.6f", (64.0/12.0)/d}')
ffmpeg -i in.mp4 -filter_complex \
  "setpts=PTS*$ratio,fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  -loop 0 out.gif
```

Speed-ups applied: `color` 1.7x, `hdr` 1.0x, `shadows` 2.0x, `temperature` 1.2x.

Unlike the other robot-free sections, these clips include the robot arm.

## Sensitivity Analysis

No new asset — the landing page reuses `docs/images/sensitivity_report_200_trails.png`, the same
figure used as the hero image in
`docs/pages/example_workflows/sensitivity_analysis/sensitivity_analysis.rst`. It shows posterior
marginals over wrist-camera displacement from a 200-episode sweep.

The companion `sensitivity_report_5_trails.png` is deliberately noisy (five episodes) to illustrate
under-sampling, so it is the wrong choice for a teaser.

To regenerate the report itself, follow the sensitivity analysis workflow page.

## Teleoperation

`docs/images/g1_galileo_arena_box_pnp_locomanip_trimmed.gif` — a re-cut of an existing gif so the
two clips in the gallery share an aspect ratio.

The source was 800x432 and its partner 800x472 (1.695:1). Cropping 34 px off each side gives
732x432 at the same ratio, leaving height, frame count and frame rate untouched:

```bash
ffmpeg -i g1_galileo_arena_box_pnp_locomanip.gif -filter_complex \
  "crop=732:432:34:0,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  -loop 0 g1_galileo_arena_box_pnp_locomanip_trimmed.gif
```

Check frames across the animation after cropping to confirm nothing important leaves the frame.

## Gallery layout

Galleries use the `image-gallery` container in `docs/_static/custom.css`, which is a flex row
driven by two custom properties:

```rst
.. container:: image-gallery

   .. image:: ./images/teaser_page/<section>/<file>.gif
```

- Default `--gallery-width: 45%` lays images out two per row — good for 2 or 4 images.
- Add the `gallery-3col` class (`.. container:: image-gallery gallery-3col`) for three per row —
  used for the 9-image galleries.
- Images in one gallery should share an aspect ratio so row heights line up.

For a code snippet beside a gallery, use a `sphinx_design` grid (the extension is already enabled)
with the `compact-code` container, as in the Automatic Object Placement section.
