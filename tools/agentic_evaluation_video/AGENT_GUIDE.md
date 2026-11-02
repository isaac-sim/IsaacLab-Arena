# Reproduce or adapt the evaluation-authoring video

Read this whole guide before recording. Work in a new project directory created
from the template. Do not overwrite the original video or copy unrelated local
experiments into the new production.

## 1. Understand the reference

The example derives from the KitchenBench peninsula scene. The task is to place
the banana into the front wooden bowl and the orange into the farther red bowl,
in either order. The displayed label is **red bowl**, without “far”. The DROID
robot uses the DROID-finetuned π0.5 policy served on port 8001.

The complete environment is `source_environment.yaml`. Four smaller YAMLs show
the authoring sequence: kitchen, wooden bowl, robot, remaining fruits and bowl.
The three movable-object stages capture five independently sampled layouts each.
Their `on` relations use `clearance_m: 0.08`, so objects drop a little after reset.
The evaluation definition has its original support relations and task semantics.
Do not copy presentation-only drop clearance into evaluation without intending it.

The two pick-and-place tasks are inside a parallel composite task. The filming
factory sets `desired_subtask_success_state = [True, True]`: both final placement
conditions must hold at completion. Preserve this when reproducing the example.
Each subtask has an 80-second timeout; the composite's default budget is their
sum, 160 simulated seconds. A failed attempt can therefore take roughly 11
wall-clock minutes to record on the reference machine. This is not a hung export.

There are two distinct clocks:

- **Visual authoring clock:** 0–53, defined by `edit_video.py` and mapped by
  `narration.py`. These values are animation coordinates, not final movie seconds.
- **Movie/audio clock:** actual Kokoro word timestamps and paragraph durations.
  `assemble_voice_over.py` builds this timeline, and `visual_time()` maps it to the
  visual clock. Retiming visuals keeps speech natural.

The terminal animation is an illustrative authoring transcript. Real scene resets
and policy inference supply the simulator footage. Do not describe the staged
terminal as evidence of a live agent session.

## 2. Prepare the project and runtimes

Run the commands in this guide from the repository root unless stated otherwise.
Use a checkout containing this kit and the repository's supported submodules.

```bash
python3 tools/agentic_evaluation_video/create_project.py outputs/my_evaluation_video
PROJECT=outputs/my_evaluation_video
```

Keep the project inside this checkout so the container mount includes it. The
creator refuses an existing destination. Use another directory for a fresh full
recording. The generated project has its own ignore rules, and `outputs/` is
already ignored by the repository.

Use these repository skills as appropriate, following the checkout's AGENTS.md:

- `skills/developer/author-environment/SKILL.md`: task, scene, relations, and
  success-condition authoring; inspect maintained examples and constructors.
- `skills/developer/dev-container/SKILL.md`: discover the clone's container and
  execute as the host user. Never hardcode a colleague's container name.
- `skills/user/setup-arena/SKILL.md`: prepare a missing Arena runtime.
- `skills/user/serve-openpi-policy/SKILL.md`: start or reuse the compatible π0.5
  server. `record.py` does not start or stop policy servers.
- `skills/user/run-experiment/SKILL.md`: evaluation configuration, canonical
  results, and report verification.

Simulation scripts must run inside the ready Arena container, through
`submodules/IsaacLab/isaaclab.sh -p`, as the host user. Host editing and speech
synthesis use a separate virtual environment. Do not install Kokoro into the
Isaac Lab environment.

The host needs FFmpeg/ffprobe, DejaVu fonts, libsndfile, and espeak-ng. On Ubuntu,
the relevant packages are `ffmpeg`, `fonts-dejavu-core`, `libsndfile1`, and
`espeak-ng`. Use the machine's normal package-installation process if missing.
The renderer uses DejaVu Sans for symbols and DejaVu Sans Mono for the terminal.

Create a Python 3.11 host environment (using uv here):

```bash
uv venv --seed --python 3.11 "$PROJECT/.venv"
VIDEO_PY="$PROJECT/.venv/bin/python"
uv pip install --python "$VIDEO_PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VIDEO_PY" -r "$PROJECT/requirements.txt"
"$VIDEO_PY" "$PROJECT/fetch_fonts.py"
```

These are reference dependency pins, not changes to Arena's dependency lockfile.
The host Python version is independent of Arena's container Python. `--seed`
provides pip for the English language model's first-run installer.
Kokoro downloads `hexgrad/Kokoro-82M` and its English language resources on first
use. Keep model caches outside Git. Speech generation is CPU-only and uses four
threads, leaving the simulator GPU available. NVIDIA Sans downloads are checked
against the recorded hashes; review an upstream change instead of bypassing the
check. A Pillow build with WOFF2/FreeType support is required for these fonts.

Verify the policy server uses the intended π0.5 DROID checkpoint and port 8001.
The OpenPI skill contains the maintained startup command and readiness checks.
A listening port alone does not identify the correct model. The reference server
used `pi05_droid_jointpos_polaris` with
`gs://openpi-assets-simeval/pi05_droid_jointpos`. Leave an existing compatible
server running after filming.

## 3. Reproduce the kitchen example

### Capture scene construction

The packaged transparent thumbnails can be reused for this exact example.
To recreate them from the installed USDs, run the optional thumbnails phase first:

```bash
"$VIDEO_PY" "$PROJECT/record.py" thumbnails
"$VIDEO_PY" "$PROJECT/record.py" captures
```

`record.py` discovers the running container whose mount points to this checkout.
If there is more than one, supply `--container NAME` after inspecting the matches.
It invokes simulation scripts with safely quoted arguments, records a log for
each phase, and stops on failures. It does not rebuild or recreate containers.

The capture phase writes 34 frames per layout: one kitchen layout, then five each
for the bowl, robot, and full scene (544 frames total). It also writes
`captures/<stage>/captures.json` with sampled object poses and simulation times,
and `capture_sources.json` tells the renderer where those frames are.

Inspect a settled frame and an early drop frame from every stage before evaluating.
The script's keys named `poses_W_xyzw` contain translation followed by an XYZW
quaternion. The scene graph is an editorial layout; the captured poses are the
evidence of actual sampled geometry.

To resume after a stage failed before writing any frames:

```bash
"$VIDEO_PY" "$PROJECT/record.py" captures --stage 03_robot
```

Existing nonempty capture folders are protected. Move an incomplete folder aside
for investigation before retrying that stage; preserve its log. Do not silently
mix captures from different source definitions. Provenance checks reject a change
of branch, commit, tracked diff, runtime container, or simulation-definition hash
between phases. Start a new project when those inputs change.

### Run three real policy evaluations

```bash
"$VIDEO_PY" "$PROJECT/record.py" evaluation
```

`experiment.yaml` contains one Run, one environment, and three episodes. The
template's `project.json` points to `reference_placements.jsonl`; the launcher
passes this replay file to the filming wrapper. `solve_relations: false` and
`resolve_on_reset: false` are intentional for this comparison. Initial placements
are replayed, but physics, observations, inference, and actions are freshly run.
Live relation solving was exercised separately by the scene captures.

Replay order is right, **front-left**, right. “Left” means shifting along the same
front edge, maintaining yaw; it does not mean moving behind the counter. The
middle reference robot pose has world X about +0.260 and Y about +0.185, with
yaw about −π/2. `front_left_sample.jsonl` preserves the corresponding sampled
layout for geometric validation. World X sign alone is not a universal left/right
test; the camera orientation matters.

The editorial camera is attached as `external_camera_2`. The policy continues to
use its normal external and wrist image inputs. Do not replace a policy input
camera just to make the film look better. The reference leaves camera resolution
at `embodiment_default`.

The underlying `run_filmed_experiment.py` also accepts `--validate-config` for
a typed-configuration preflight inside the container. Supply the same
`--experiment_config`, `--output` (a new empty directory), and optional
`--replay-placements` arguments. This only loads the configuration and checks its
file references; it does not build/reset the scene or prove policy execution.

Wait for all episodes and report generation. Inspect:

- `evaluation/arena_experiment_result.json`: Run status `completed`, correct
  `git_stamp`, rebuilds and episode outcomes.
- `evaluation/index.html`: navigable report and linked per-Run pages.
- `evaluation/<run>/episode_results_rebuild0.jsonl`: native progress, events,
  sampled placement, and true success/failure.
- `evaluation/<run>/robot-cam-...-external_camera_2_rgb-episode-N.mp4`: editorial
  camera clips, one for each episode.

`completed` means the evaluation ran; it does not mean the policy succeeded.
Never turn a failed episode's status box green or drop a failure without disclosing
selection. The wrapper explicitly checks final Run status after writing the report.

### Extract and validate the clips

```bash
"$VIDEO_PY" "$PROJECT/summarize_evaluation.py" "$PROJECT/evaluation"
"$VIDEO_PY" "$PROJECT/prepare_rollouts.py" "$PROJECT/evaluation" \
  --include-failures --episodes 0 1 2 --limit 3 --frames-dir rollout_frames
```

The first command cross-checks canonical results against native episode records.
The second extracts all three attempts at 6× their recorded-video rate, using
FFmpeg's `setpts` and 30 fps output. It records the selected sources and outcomes
in `rollouts.json` and refuses to overwrite an existing frame folder. The source
camera rate is read from each MP4 rather than assumed. Playback rate here refers
to the recorded video, which need not equal simulated wall time.

For this reference, use exactly the listed three episodes in order. Robot-side
labels in `project.json` describe that order; the validator additionally checks
the middle episode against its actual native placement. For a different selection,
update the labels and geometric expectations rather than merely changing text.

### Generate fixed-speed narration and render

```bash
"$VIDEO_PY" "$PROJECT/generate_voice_over.py"
"$VIDEO_PY" "$PROJECT/assemble_voice_over.py"
"$VIDEO_PY" "$PROJECT/edit_video.py" --stills
```

Inspect `narration_review/`, `preview_52.jpg`, and `preview_evaluation_layout.jpg`.
Listen to `voice_over_readthrough.wav` and then `voice_over.wav`. The readthrough
is continuous speech; the latter includes gaps for the footage. The voice is
Kokoro **af_heart**, American female, fixed speed **1.0**. The reference speech
was about 144 words/minute, near the 140 wpm target.

`generate_voice_over.py` caches paragraph WAVs and timestamp JSON under
`audio/fixed_speed/`. Pronunciation substitutions include π0.5 as “pie zero point
five” and USD as spelled letters. Word timestamps are mapped back to the displayed
script. `assemble_voice_over.py` schedules the unmodified takes and may slow the
rollout footage to leave enough room for its narration. It never stretches speech.
Long failed attempts can leave a quiet interval before the closing sentence.
Do not speed individual voice segments up or down to force a fixed video length.

When the stills and speech are correct:

```bash
"$VIDEO_PY" "$PROJECT/edit_video.py"
"$VIDEO_PY" "$PROJECT/validate_video.py"
```

The output is `kitchenbench_split_bowls_authoring.mp4`, 1920×1080, 30 fps, H.264,
with one AAC 48 kHz mono track. The exporter replaces that filename inside the
working project; copy a prior revision aside before iterative exports.

`validate_video.py` decodes the full movie, checks the provenance and captures,
matches outcomes against native results, checks the front-left replay, verifies
caption width and timing, natural speech durations, reference flashes, opening
panels, prompt flash, and final status colors. It creates `video_validation.json`,
`review_contact_sheet.jpg`, `poster.jpg`, and `encoded_review/`. Inspect those
images and watch the movie; automated pixel checks do not assess every motion.
For an editing-only check of reused footage, `validate_video.py --media-only`
explicitly skips fresh-source provenance checks and records that limitation in
its JSON. Do not use that mode to claim a new simulation run.

The only rendered caption is the smaller 32 px line at the bottom. The editable
SRT is `subtitles/narration.srt`. Do not put a same-basename SRT beside the MP4:
players may automatically display it as a second copy on top of burned captions.

## 4. Adapt to your colleague's task

First inspect her actual environment/Experiment Definition, task predicates,
embodiment, available policy, and result artifacts. Reuse those semantics. Ask
only for information that cannot be found in the checkout or connected inputs.
Do not assume a new task can be handled by relabeling these two fruit nodes.

Create a new project, then make these changes before recording:

| Concern | Files and decisions |
| --- | --- |
| Full task | Replace `source_environment.yaml` or adapt `FilmEnvironment.build()` to the existing Python factory. Preserve predicates, contacts, timeout, and episode completion semantics. |
| Composite requirements | Update `[True, True]` in the wrapper and `expected_subtasks` in `project.json`. An atomic task needs a different final-state setup; do not attach composite-only fields blindly. |
| Authoring stages | Edit the four stage YAMLs, their `project.json` stems/counts, and the capture helper's `NoEmbodiment` stage list. Keep the early illustrative drops separate from evaluation behavior. |
| Initial layouts | Either prepare three native placement records for the new assets, or set `replay_placements` to `null` and enable `solve_relations`/`resolve_on_reset` in the Experiment. Never replay the kitchen's poses into unrelated assets. |
| Left-side requirement | It belongs to this kitchen film. For another scene, replace it with an appropriate geometric check, or disable `require_front_left` and set truthful `rollout_sides`. Remove obsolete reference-placement files from that project's deliverables. |
| Policy | Set the correct policy class/config, server, embodiment adapter, and checkpoint. The helper currently assumes a DROID-style camera configuration; adapt it for other embodiments. |
| Camera | Edit `camera.json` eye/target/focal length; check reach, gripper visibility, bowl interiors, and both robot positions. `film_camera.py` uses active world-to-camera pose notation and XYZW quaternions. |
| Thumbnails | Update `render_asset_thumbnails.py` USDs, scale, references, and posing. Re-render isolated assets; use transparent PNGs, not cutouts from a scene screenshot. |
| Terminal | Update `PROMPT` and `COMMANDS` in `edit_video.py`. Keep the prompt as input and the other lines as agent replies. |
| Graph and task cards | Update `components()`, node keys, labels, edges, and `rollout_board()` in `edit_video.py`; see the visual specification below. The graph is manually laid out, not generated automatically from YAML. |
| Speech and captions | Edit `VOICE_OVER.md` and `PARAGRAPHS` in `narration.py` together. Preserve section/paragraph structure; the scripts assert exact agreement. |
| Spoken references | Update `REFERENCES` in `narration.py` and `REFERENCE_BOUNDS`/`ROLLOUT_REFERENCE_BOUNDS` in `edit_video.py` so mentions flash the correct visible box. |
| Validation | Adapt the kitchen-specific caption fragments, visual-clock anchors, placement checks, criterion counts, and review times in `validate_video.py`/`prepare_rollouts.py`. Keep generic audio/decode/outcome checks. |

The current renderer expects **four authoring stages, five samples for each object
stage, two task cards, and three rollouts**. If the new task needs another count,
change the layout and validation as well as the metadata. Changing `project.json`
alone does not redesign the renderer.

`prepare_rollouts.py` interprets reference pick-and-place event indices 1 as lift
and 2 as placement. Other task types may have different predicates; adapt those
checks to their actual event schema. The convenience summary currently expects
one rebuild and the reference policy. Extend it explicitly for another evaluation
shape rather than using a report that silently omits rebuilds.

### Narration editing procedure

1. Finalize prose first. Aim around 140 words/minute; the reference has 225 words.
2. Split each paragraph into short caption phrases in `PARAGRAPHS`. Give each
   authoring phrase its visual start/end coordinates. Evaluation phrases use
   `(None, None)` and show rollouts. Keep the final outcomes paragraph last.
3. Update `REFERENCES` with exact caption fragments, spoken phrases, and targets.
   The code finds exact word matches; a paraphrase does not automatically match.
   Delay a highlight until the referenced asset/card has appeared.
4. Ensure `script_text()` equals `VOICE_OVER.md`. Running `narration.py` directly
   rewrites the Markdown from `PARAGRAPHS`; use that only intentionally.
5. Run generation, assembly, still rendering, full export, and validation in that
   order. Changed speech text invalidates the timing hash. Regenerate audio before
   retiming visuals. Never reuse stale timestamps after a text edit.
6. Listen for pronunciation and pacing; inspect every caption for clipping and
   each flash for the right target. The word count of expanded pronunciations must
   still align back to displayed words; update `PRONUNCIATIONS` if needed.

## 5. Preserve the visual specification

- White background, NVIDIA green `#76b900`, light-gray inner cards `#eeeeee`,
  restrained gray text, NVIDIA Sans. Terminal text is DejaVu Sans Mono.
- Fixed two-line title: **Isaac Lab - Arena**, 52 px bold black; below it
  **Agentic Evaluation Authoring**, 36 px gray. The title never tracks camera or
  panel movement.
- Three outer panels: **AGENT** with Codex icon and “Benchmark author”; **Isaac Lab
  - Arena** with Isaac Lab icon and “Evaluation description framework”; **Isaac
  Lab** with Isaac Lab + Omniverse icons and “GPU-accelerated simulator”.
- Curved green author/run arrows between the panels. Bolder panel outlines.
- Task, scene, and policy cards visible but empty from the beginning. No blank
  opening screen. Populate them as narration introduces each component.
- Codex-like terminal inside a macOS window: traffic-light buttons, compact
  monospaced lines, gray input prompt, bullet replies. No line numbers and no
  extra “Open AI Codex” line. Keep the full transcript visible without scrolling.
- Reference prompt: “Design an evaluation environment in a kitchen where two
  pieces of fruit must be placed into two bowls.” Flash its gray input box when
  the narration reads it.
- Scene graph: banana then orange on the first row; wooden bowl then red bowl
  below, aligned in left/right columns. Robot below; kitchen contains dashed
  island-top and floor references. These are references into the kitchen USD,
  not independently added physical assets or proof the kitchen itself is on the
  floor. Show the relevant support/alignment relation labels; omit the extra
  fruit/bowl/robot constraint-edge clutter.
- Small isolated asset images in the graph nodes. Add remaining assets in one
  burst without a group container. Sample five times quickly; flash on resets.
- Composite task box containing two pick-and-place cards added quickly in
  succession, with visible links to their graph assets. Add a π0.5 policy card
  with the Physical Intelligence logo before evaluation.
- Evaluation screen: fixed task definition and a separate policy card with a
  smaller logo on the left; rollout on the right. Three outcome boxes along the
  bottom, gray while pending, green for success, red for failure. No “6xplayback”
  text. No motion of the task/policy text between runs.
- One smaller caption line at the bottom, synchronized to speech. Flash the task,
  scene, policy, prompt, and other named components as they are mentioned.
- Do not reintroduce discarded decorative text such as “A scene definition becomes
  a simulation”, “kitchen anchors”, “Release → Settle”, or a changing top subtitle.

## 6. Troubleshooting and handoff

| Symptom | Check |
| --- | --- |
| `PositionLimitsBox` rejects `x_min` | Current schema uses `x_limits: [minimum, maximum]` and analogous Y/Z pairs. Preserve numerical bounds when migrating. |
| Isaac Lab/Warp startup error after pulling main | Confirm the running container uses the rebuilt image and current checkout, not an old snapshot. Follow setup/container skills; do not downgrade or patch Arena merely to reuse stale footage. |
| No policy connection or wrong behavior | Verify server model/config, host-network port, embodiment adapter, and expected observation cameras. |
| Long evaluation with little log output | Check native episode records and growing camera files. A 160-second simulated timeout can take minutes; keep the full budget unless the user changes it. |
| A Run failed but some clips exist | Read the exception log and canonical status. Preserve the attempt; do not claim the pipeline passed or summarize it as a completed benchmark. |
| Double captions | Remove the auto-loaded same-basename SRT from beside the MP4; keep the editable SRT under `subtitles/`. |
| Unnatural voice speed changes | Regenerate at one Kokoro speed; retime video around the actual takes. No per-segment `atempo` or time stretching. |
| eSpeak reports a missing build-machine `phontab` path | Deeply nested virtual-environment paths can trigger this. The generator copies eSpeak's installed data to a short temporary directory while synthesizing speech; keep that setup when adapting the script. |
| Stale text/timing assertion | Synchronize Markdown and caption phrases, regenerate voice, then reassemble. |
| Robot behind counter | Compare world poses, yaw, and camera view, not an invented “left” label. |
| Validator fails after task adaptation | Inspect the specific assertion. Update genuinely task-specific expectations; do not delete generic truthfulness/audio/media checks to force a pass. |

Hand back the MP4 path, duration, actual policy outcomes, environment and Experiment
paths, branch/commit, report link, validation JSON, contact sheet, and any remaining
limits. Distinguish preserved assets from newly recorded simulation footage and
pipeline execution success from policy success. Retain the source project and
provenance so the next agent can reproduce or revise it.

For a PR of this kit or a new template, include source code, configuration,
instructions, and small required visual assets. Keep generated movies, captured
frames, episode output, audio, virtual environments, and model caches out of Git.
