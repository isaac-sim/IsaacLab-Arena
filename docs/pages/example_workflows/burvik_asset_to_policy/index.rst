From reference assets to an Arena assembly policy
=================================================

This worked example follows one interactive Codex session on October 12–13,
2026: IKEA BURVIK reference images and assembly instructions became six USD
parts, an Arena environment with five ordered assembly tasks, and a state-based
bimanual YAM policy. Blender MCP was used for asset authoring, and the
``author-environment`` skill guided Arena integration. The resulting environment
is a Python factory named ``burvik_assembly_yam``.

The three checkpoints below separate asset physics, task verification using
scripted part motion, and robot execution. Their measurements come from saved
successful validation reports; the animations show those same runs. The second
checkpoint has **no robot manipulation**: the YAM is present but parked. The
third uses actual gripper contacts and robot pose actions.

.. list-table:: Recorded checkpoints
   :header-rows: 1
   :widths: 25 25 25 25

   * - Checkpoint
     - Driver
     - Recorded sequence
     - Result
   * - Standalone asset check
     - Scripted forces / part motion
     - 104.96 s
     - Loose and assembled stability passed
   * - Arena task check
     - Scripted part motion; YAM parked
     - 81.38 s; task completed at 74.875 s
     - All five ordered subtasks passed
   * - Robot IK policy
     - Absolute TCP poses and gripper commands
     - 267.54 s
     - All five subtasks and current final conditions passed

The original MP4s are 1280 × 720 at 24 fps and remain local. The embedded GIFs
cover each complete sequence at reduced resolution and 3–4 fps, with 4× playback
for the first two and 8× for the robot. Each GIF is below 2 MB, satisfying both
the requested 10 MB limit and the repository's stricter commit limit. PNGs retain
the original resolution. These documentation images are the only committed
rendered outputs; USDs, meshes, textures, and full recordings remain local.

.. list-table:: Committed animations (decimal MB)
   :header-rows: 1

   * - File
     - Resolution
     - Target fps
     - Size
   * - ``standalone.gif``
     - 400 × 226
     - 4
     - 1.178 MB
   * - ``arena.gif``
     - 480 × 270
     - 4
     - 1.320 MB
   * - ``robot_ik.gif``
     - 400 × 226
     - 3
     - 1.711 MB

Prerequisites and local inputs
------------------------------

Use an Arena checkout with the BURVIK implementation (introduced in commit
``b6a56ae34c``), a running checkout-specific Arena container, Blender with
``bpy`` / ``mathutils`` / ``pxr``, and the local YAM USD payload. The original
session used the ``isaaclab_arena:latest`` image. The image tag is mutable;
the saved recordings are evidence of that session, not a guarantee of identical
results on another image or GPU.

The original reference material was in ``~/Downloads/ikea``. Its dimensions and
estimated physical properties were distilled into
``artifacts/ikea_burvik/specification.json``. Asset generation scripts contain
the resulting geometry and do not require resubmitting the original prompts.

The environment opens these seven local USD entry points:

.. code-block:: text

   artifacts/ikea_burvik/assets/tray.usd
   artifacts/ikea_burvik/assets/leg_1.usd
   artifacts/ikea_burvik/assets/leg_2.usd
   artifacts/ikea_burvik/assets/leg_3.usd
   artifacts/ikea_burvik/assets/leg_4.usd
   artifacts/ikea_burvik/assets/handle.usd
   .cache/tool_hanging_mdl_patch/assets/industrial__i2rt_yam/i2rt_yam_default.usda

The last file is instantiated twice. Keep its referenced ``i2rt_yam/Payload``
layers and materials, and the BURVIK textures, beside their entry points.
The bench, floor, robot pedestals, and handle presentation rails are generated
in code. The YAM implementation lives in ``isaaclab_arena/embodiments/yam``;
the environment does not require the old ``isaac_cap`` Python package.

On the host, from the repository root, discover the container and define a
helper for the reproduction commands below. Runtime commands execute as the
host user inside the container:

.. code-block:: bash

   ARENA_CONTAINER=$(docker ps \
       --filter "volume=$(git rev-parse --show-toplevel)" \
       --format '{{.Names}}' | head -1)
   test -n "$ARENA_CONTAINER"
   arena() {
       docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
           "cd /workspaces/isaaclab_arena && $*"
   }

Use fresh output directories for new runs. Asset generation overwrites local
generated assets, so preserve any manually edited assets first. The final code
includes fixes developed across all three stages; rerunning an earlier demo
with that code is a regression check, not a byte-identical historical replay.

1. Standalone simulation and asset check
----------------------------------------

User prompts
~~~~~~~~~~~~

The prompts are reproduced verbatim, including their original spelling.
The initial capability check was: ``check if you have access to blender mcp``.

.. raw:: html

   <blockquote><p>checkout the ~/Downloads/ikea folder containning visual, dimension and assembly instruction. Use the blender mcp to create physicsally accurate 3D parts with good visual aligment. all parts should have physics collision that is ready to be used with isaac sim physx or newton. for the screws on the leg and matching screw holds in the table top, simplify to remove threading, and the attachedment should be just pushing the simplify screen rod into the hole. export the assets to USD, then import them in isaac sim using isaaclab_arena:latest. verify in isaac sim that the parts are stably placed on a table top, and once assembled, the finished table also stay physically stable. show me a video of the full assemblying process in isaac sim, using scriptted force or displacement on the parts during the assbly process</p></blockquote> <!-- codespell:ignore -->

Result and visuals
~~~~~~~~~~~~~~~~~~

Six independent rigid bodies were exported: a 38 cm tray, four legs, and a
handle. Smooth 8 mm pins replace threads; the leg and handle bores are 9.2 mm
and 10.8 mm respectively. Collision geometry preserves the open holes. The
standalone demo seats parts using scripted motion, creates part-to-part
retention joints, turns the table upright, and releases it under gravity.
No part is fixed to the world.

.. figure:: /images/burvik_workflow/standalone_loose.png
   :alt: Six separate BURVIK rigid parts resting on the workbench.
   :width: 720

   Loose parts after gravity settling.

.. figure:: /images/burvik_workflow/standalone.gif
   :alt: Complete standalone part assembly, turning, release, and stability check.
   :width: 640

   Complete standalone simulation, 4× playback.

.. figure:: /images/burvik_workflow/standalone_assembled.png
   :alt: The completed BURVIK table standing freely on the workbench.
   :width: 720

   Released assembly at the end of the standalone run.

``results/validation.json`` reports ``passed=true``, all bodies dynamic, and
zero world fixed joints. Maximum part drift was 0.010 mm over the two-second
loose-part check and 0.0143 mm over the five-second assembled check. After a
1 N lateral disturbance lasting 0.25 s, a further three-second check passed.
The full local video is ``results/burvik_assembly.mp4`` under
``artifacts/ikea_burvik``.

Reproduce
~~~~~~~~~

Run the Blender commands on the host, then the standalone simulation in the
container. These are the maintained scripts corresponding to the original
Blender MCP operations:

.. code-block:: bash

   blender --background --python tools/burvik_assets/build_blender.py
   blender --background --python tools/burvik_assets/package_assets.py
   arena /isaac-sim/python.sh tools/burvik_assets/simulate_assembly.py --video
   arena /isaac-sim/python.sh tools/burvik_assets/audit_delivery.py

When executing through Blender MCP, set ``__file__`` to the script's absolute
path before executing it so sibling helpers resolve. ``build_blender.py`` also
calls the handle recovery exporter. ``triangulate_visuals.py`` can refresh
visual meshes from an existing ``burvik_source.blend``; see
``tools/burvik_assets/README.rst`` for that command.

Changes that established this checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Converted references into meter / kilogram / Z-up USDs with separate rigid
  bodies, explicit estimated mass and inertia, and convex collision proxies.
  The tray has 237 proxies, each leg four, and the handle 41; hole interiors
  remain open for insertion.
* Repaired the handle export and triangulated visual meshes to preserve its
  appearance in Isaac Sim. Packaged relative USD dependencies and materials.
* Added smooth pins and lead-ins, controlled insertion, seating-triggered
  retention, free-standing checks, and a disturbance check. Asset audits check
  composition, collider counts, triangle meshes, and final foot support.

Recorded authoring time was **1 h 11 min 20 s**, including debugging and
simulation waits. Usage was **14,031,867 total tokens**, including
13,754,240 cached input tokens; detailed accounting is below.

2. Arena environment and scripted task verification
---------------------------------------------------

User prompts
~~~~~~~~~~~~

.. raw:: html

   <blockquote><p>use the arena author-environment skill, set up an arena environment with a bimanual yam robot (bring it from the main_old branch's isaac-cap environment folder), use the above table and ikea bulvik parts as scene. make sure the parks are placed reachable form the yam. create a long horizon composite task representing the bulvik table assembly task (should be 5 steps, 4 for instally the leg and 1 for installing the holder). reuse existing tasks/predicates if possible but can also create new ones if necessary. after env created, turn your above demo into a scripted  demo script using the newly created arena env, and ensure that the task are completed successfully as expected.</p></blockquote> <!-- codespell:ignore -->
   <blockquote><p>run the demo and record demo video from vierport camera and/on from the yam robot top camera</p></blockquote> <!-- codespell:ignore -->

Result and visuals
~~~~~~~~~~~~~~~~~~

The asset scene became ``BurvikAssemblyEnvironment`` with a reachable layout
and a sequential composite task: **leg 1 → leg 2 → leg 3 → leg 4 → handle**.
The existing part-motion demo now steps the actual wrapped Arena environment.
The YAM remains parked; dynamic part guidance is removed for the final
five-second stability check.

.. figure:: /images/burvik_workflow/arena.gif
   :alt: Scripted part assembly in Arena while both YAM arms remain parked.
   :width: 640

   Actual viewport capture from the Arena task check, 4× playback.

.. figure:: /images/burvik_workflow/arena_assembled.png
   :alt: Completed Arena assembly on the bench with both YAM arms parked.
   :width: 720

   Final viewport image after scripted part guidance was released.

``arena_camera_recording/validation.json`` records all five ordered events,
``final_subtask_completion=[[true, true, true, true, true]]``, and overall pass.
Task completion occurred at 74.875 simulated seconds. Maximum drift during
the five-second released check was 0.000434 mm; measured upward bench support
was 24.81 N. The local recordings are
``arena_camera_recording/burvik_arena_viewport.mp4`` and
``arena_camera_recording/burvik_arena_top.mp4``.

Reproduce
~~~~~~~~~

.. code-block:: bash

   arena /isaac-sim/python.sh \
       -m isaaclab_arena_environments.burvik_assembly.scripted_demo \
       --video --record-cameras viewport top --check-reachability \
       --output artifacts/ikea_burvik/workflow_arena
   arena /isaac-sim/python.sh tools/burvik_assets/audit_arena_export.py \
       artifacts/ikea_burvik/workflow_arena

The output includes both MP4 streams, final PNGs, ``validation.json``, a saved
trajectory, and an exported USD scene. The optional reachability check tests
positional TCP targets against joint limits; it does not certify robot grasp
orientations or collision-free paths. Those were exercised in checkpoint 3.

Updates and fixes since checkpoint 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Registered the environment and imported the bimanual YAM from ``main_old``
  commit ``c8d04e2199b86abbbb301bb22cec0effd83c4e63``. Repositioned parts within
  its workspace while retaining their physical scale and bench dimensions.
* Reused ``CompositeTaskBase``, ``TrueForConsecutiveStepsCfg``, upward-support
  checks, dropped-object termination, and Arena's progress recorders.
  Added ``InstallBurvikPartTask`` because the existing assembly task checked
  root proximity rather than actual socket alignment and retention.
* Required tray-relative pose, an enabled connection, correct standing height,
  upright orientation, low motion, and bench support. Connections reset per
  environment. This prevents a nearby, unsupported, or incorrectly oriented
  part from satisfying the task.
* Added the handle's 1.5 mm stand-off to clear its tube shoulders. Used Arena's
  physics stepping and real viewport / top-camera capture, with framing that
  includes the completed handle.

Recorded authoring and recording time totaled **1 h 13 min 49 s** across two
requests. Usage was **20,622,391 total tokens**, including 20,086,400 cached
input tokens.

3. Bimanual robot IK policy
---------------------------

User prompts
~~~~~~~~~~~~

The control-interface question was:
``does the arena yam robot accepts IK control``.

.. raw:: html

   <blockquote><p>update yam to support IK controller, use it for the burvik env. now, write a scripted state-based policy  that sends pose control for the bimanual yam to solve the full task</p></blockquote> <!-- codespell:ignore -->

Result and visuals
~~~~~~~~~~~~~~~~~~

The registered ``burvik_yam_assembly`` policy returns robot TCP pose and
gripper actions. It reads simulation state to guard pickup, grasp, lift,
alignment, insertion, retention, release, and bimanual turning. It installs
all four legs, turns the assembly sideways, inserts the handle, regrasps,
turns the table upright, places it, opens both hands, and waits for current
Arena success. It does not teleport parts to solve the task.

.. figure:: /images/burvik_workflow/robot_ik.gif
   :alt: YAM arms picking and inserting all parts, turning and releasing the assembled table.
   :width: 640

   Full successful physical rollout rendered through the YAM top camera,
   8× playback. Rendering replays saved states without advancing physics;
   task success was measured in the preceding physics run.

.. figure:: /images/burvik_workflow/robot_turn.png
   :alt: Both YAM arms physically gripping opposite legs during a coordinated table turn.
   :width: 720

   Bimanual turning at 135 seconds of the recorded rollout.

.. figure:: /images/burvik_workflow/robot_assembled.png
   :alt: The completed table viewed from the top with both grippers open after release.
   :width: 720

   Final released state. The validation report establishes support and settling,
   which the top view alone cannot demonstrate.

``yam_ik_assembly_v10/validation.json`` reports ``passed=true``, all five
completion flags true, and ``current_task_success=true``. The released table
remained supported for 5.3375 s with maximum part drift **0.147 mm**, maximum
seating position error **0.334 mm**, and upward bench support **25.68 N**.
The local MP4 is ``yam_ik_assembly_v10/yam_top.mp4``; ``rendering.json`` identifies
it as playback of that successful physics trajectory.

Reproduce
~~~~~~~~~

.. code-block:: bash

   arena /isaac-sim/python.sh \
       -m isaaclab_arena_environments.burvik_assembly.robot_demo \
       --output artifacts/ikea_burvik/workflow_robot --max-seconds 600
   arena /isaac-sim/python.sh \
       -m isaaclab_arena_environments.burvik_assembly.render_rollout \
       artifacts/ikea_burvik/workflow_robot

The first command must finish with a successful ``validation.json`` before the
renderer accepts its trajectory. Add ``--video`` to ``robot_demo`` for live
top-camera capture instead; it costs additional simulation runtime.
``--stop-after-leg 1`` is only a partial diagnostic, not full-task verification.

Updates and fixes since checkpoint 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added ``ik_pose`` (16 actions) and ``ik_delta`` (14 actions), preserving joint
  control. An absolute action contains left TCP ``xyz, qx, qy, qz, qw``, left
  gripper, then the same for the right arm. Each pose is in that arm's base
  frame; grippers use 0=open, 1=closed. BURVIK now defaults to ``ik_pose``.
* Corrected the Jacobian for the rotated TCP offset, handled constrained-base
  entries in gravity buffers, normalized quaternions, rejected invalid actions,
  and bounded joint targets. Damped least-squares IK and gravity compensation
  through drive targets retain effort limits.
* Raised arm bases to 0.90 m and added handle presentation rails for physical
  grasp clearance. Adjusted handle jaw angle and transfer waypoints for wrist
  limits, tube clearance, and two-pin alignment. Insertion advances only below
  0.3 mm lateral and 0.2 degree angular error.
* Replaced independent loaded wrist motion with a common table trajectory and
  measured grasp transforms. Ramped grippers, shortened final grasp lever arms,
  and added bounded payload-droop compensation. Tuned contact solver settings
  and servo gains to sustain loaded bimanual grasps.
* Removed uncalibrated connector break limits that failed during the final turn.
  Seated pins use ideal, nonbreaking part-to-part fixed joints. This is a
  push-fit retention approximation, not a measured friction or fracture model.
* Corrected settling checks to use measured pose change when the solver reports
  angular-velocity residuals without corresponding movement. Full quaternion
  rotation still counts as motion. After release, the policy waits for Arena's
  existing consecutive-step success condition instead of finishing when past
  milestones alone are complete; seating and settling limits were not relaxed.

Recorded development time was **15 h 42 min 9 s**, covering repeated failed
attempts, diagnosis, simulation, rendering, and waits. Usage was
**127,957,154 total tokens**, including 125,439,872 cached input tokens.
This is not the runtime of one rollout.

The complete recording predates the final move of YAM Python modules into
``isaaclab_arena/embodiments/yam``. After that move, seven focused tests passed
and a physical first-leg diagnostic passed with 0.295 mm seating error. The
full policy was not rerun after that packaging-only change. The reproduction
commands above use the final module locations.

Time and token accounting
-------------------------

These are observed session counters, not estimates or a billing calculation.
Elapsed time sums the recorded request durations, including tool execution,
debugging, waits, and pauses. Stage 2 combines the environment-authoring and
camera-recording requests. Capability questions, gaps between requests, Git
operations, USD-list questions, and this documentation work are excluded.

Input tokens count repeated context sent to model calls; cached input is a
subset of that input. Output includes reasoning tokens. Total equals input
plus output, so neither cached input nor reasoning is added again. The counters
were checked against the sum of individual response usage records. No dollar
cost or active human/GPU time can be inferred from this table.

.. list-table:: Usage across all attempts for each checkpoint
   :header-rows: 1

   * - Checkpoint
     - Elapsed request time
     - Input tokens
     - Of which cached
     - Output tokens
     - Total tokens
   * - Standalone
     - 1:11:20.328
     - 13,967,791
     - 13,754,240
     - 64,076
     - 14,031,867
   * - Arena
     - 1:13:49.203
     - 20,523,310
     - 20,086,400
     - 99,081
     - 20,622,391
   * - Robot IK
     - 15:42:08.700
     - 127,337,985
     - 125,439,872
     - 619,169
     - 127,957,154

The :download:`checkpoint evidence summary <evidence.json>` contains exact
durations, counters, selected validation metrics, and hashes of the source
reports. Raw session logs are not included.

Recreate the documentation media
--------------------------------

With the original three recording directories available, run on the host:

.. code-block:: bash

   python tools/burvik_assets/make_workflow_media.py

For fresh outputs, pass ``--standalone``, ``--arena``, and ``--robot`` directory
overrides. The helper requires host FFmpeg / FFprobe and Python 3.11 or newer;
it copies the PNGs, extracts the robot turning frame, and converts each complete
MP4 to a looping GIF. It downscales, samples at 3–4 fps after acceleration, and
reduces the palette as needed, failing if any output reaches 2,000,000 bytes.
The :download:`media manifest </images/burvik_workflow/manifest.json>` records
source hashes, durations, frame rates, output dimensions, and exact file sizes.

Scope of the result
-------------------

The demonstrated backend is **Isaac Sim / PhysX**. Newton compatibility and
dynamic retention were not validated. Dimensions follow the supplied references;
mass, inertia, friction, servo gains, and connector properties are estimates or
simulation tuning, not hardware calibration. The policy uses simulator state,
one environment, fixed grasps, and a fixed layout; it is not a vision policy or
a general collision planner. The three checkpoints establish a working workflow
and successful runs under those conditions.
