Record and Replay Clutter Layouts
==================================

Use ``ClutterOn`` to release objects above a fixed support, let physics settle
them, and save accepted layouts for later resets. The result is a JSONL file of
complete scene root poses, including the robot root. The workflow is:

**Solve release poses → step physics → check acceptance → save JSONL → replay.**

This guide starts with one layout, then shows batch recording and a bowl example.
See :ref:`ClutterOn settings <clutter-on-relation>` for relation parameters and
:doc:`recording` for the shared recorder API and ordinary placement examples.

Before You Start
-----------------

Complete :doc:`../../quickstart/installation` and select your
:ref:`Arena runtime <placement-recording-runtime>`. Run every command from the
repository root in that runtime's shell. Asset downloads must be accessible;
the first run can take longer while assets load.

These recipes explicitly use **PhysX on CPU**. ``--viz kit`` requires a graphical
display, including display forwarding in a remote session. For headless
recording, replace ``render=true --viz kit`` with ``render=false --viz none``.
Keep the backend, asset geometry, and reset configuration consistent on replay.

The supplied scenes already meet the following requirements. When adapting a
scene, check them before recording:

* Supports are both ``IsAnchor`` assets and physically static or kinematic.
  Marking an asset as an anchor alone does not stop it moving in simulation.
* Clutter members are dynamic rigid objects with gravity enabled. Each has one
  ``ClutterOn`` spatial relation to a fixed support.
* Other placement relations have already been resolved to fixed anchors.
  This collection path does not solve a movable support or an ordinary fixture
  around the clutter during settling.
* Supports are upright, with only quarter-turn rotations about world Z.
  Object sets and clutter reachability requirements are unsupported.
* Recorded scene roots have compatible pose resets. Joint states and other
  randomized properties are not saved.

See :ref:`clutter-recording-preflight` for exact restrictions and remedies.
Use a **new output path** for each recording; existing files are not overwritten.

.. _tool-clutter-on-a-table:

Record One Table Layout
------------------------

The maintained scene releases three hammers and a clamp above an office table.
Its relation definitions are:

.. literalinclude:: ../../../../isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml
   :language: yaml
   :start-at: relations:
   :end-before: task:

Record one accepted layout:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       output=outputs/clutter/tools_on_table.jsonl \
       num_envs=1 min_layouts=1 layouts_per_env=1 max_batches=15 seed=42 \
       settle.num_steps=480 \
       +settle.validators.support_containment.minimum_resting_heights_m.office_table_background=0.5306 \
       presets=physx render=true --device cpu --viz kit

The table has a beveled top. ``0.5306`` is its measured surface height in the
scaled table-local frame, in metres. ``office_table_background`` is the runtime
scene key, not the YAML node ID ``table``. A leading ``+`` adds this new key to
Hydra's configuration dictionary; existing fields such as ``settle.num_steps``
use plain ``=``.

The command attempts at most 15 reset-and-settle batches. It stops after one
layout passes the required solver checks and all applicable enabled
post-physics checks. The acceptance criteria are described under
:ref:`clutter-recording-checks`.

.. figure:: ../../../images/clutter/release.png
   :width: 640px
   :alt: Three hammers and a clamp suspended above the office table.

   An example solver layout before settling. Sampled positions vary by run.

.. figure:: ../../../images/clutter/settled.png
   :width: 640px
   :alt: The same hammers and clamp resting on the table after settling.

   The corresponding accepted layout after settling. Acceptance is determined
   by the recorded checks, not by the image alone.

.. _recording-and-replay-notes:

Check the Output
~~~~~~~~~~~~~~~~~

On success, ``outputs/clutter/tools_on_table.jsonl`` contains exactly one line.
The console reports progress and then a line of this form:

.. code-block:: text

   [recording] batch <batch>/15: 1/1 collected
   Saved 1/<attempted> accepted layouts: outputs/clutter/tools_on_table.jsonl

The supplied PhysX scenes can also emit :ref:`kinematic-body diagnostics
<clutter-physx-diagnostics>` during setup and reset. Read that explanation when
checking the log; it applies to those specific messages, not other runtime errors.

Inspect the recorded root names and validation reports:

.. code-block:: bash

   python - <<'PYTHON'
   import json
   from pathlib import Path

   path = Path("outputs/clutter/tools_on_table.jsonl")
   records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
   assert len(records) == 1, f"Expected one layout, got {len(records)}"
   placement = records[0]["variations"]["scene.relation_placement"]
   print("Source:", placement["source"])
   print("Roots:", sorted(placement["poses"]))
   print(json.dumps(placement["validation"], indent=2))
   PYTHON

``source``, ``poses``, and ``validation`` all belong to the
``scene.relation_placement`` variation. An applicable post-physics check must
have ``passed: true``. ``passed: null`` denotes a skipped check with a reason.
Use :doc:`qualification` to check every pose, report, and replayed root.

For larger targets, exhausting the batch budget can produce a **partial file**.
With zero accepts, **no file** is written. A shortfall is logged as an error but
the command returns normally: neither exit status zero nor file existence proves
that the target was reached. Inspect the accepted count and
:ref:`rejection summary <recording_rejection_summary>` before changing settings.
More batches give more attempts; they do not relax the checks.

Replay in the Viewer
~~~~~~~~~~~~~~~~~~~~~

Load the same scene and the file just recorded:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       --placement_layouts outputs/clutter/tools_on_table.jsonl \
       --num_envs 1 --device cpu --viz kit

The viewer starts with the recorded arrangement. These examples use ``NoTask``:
they load the first layout and do not trigger episode resets. Close the viewer
or press Ctrl-C to exit. Physics continues after reset, so a visible match is
only a visual check. :doc:`qualification` compares poses numerically across
multiple resets, including queue wraparound, without a viewer.

For policy evaluation, configure the same scene in an Experiment and pass this
file as its placement layouts. Follow the :ref:`evaluation replay instructions
<placement-recording-replay>` and :ref:`placement-replay-configuration`;
an experiment using a different scene's root names cannot replay this file.
The ``NoTask`` examples demonstrate placement, not task success.

Record a Batch
---------------

After the one-layout example works, request ten layouts across four environments.
This headless command writes a separate file:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       output=outputs/clutter/tools_on_table_batch.jsonl \
       num_envs=4 min_layouts=10 layouts_per_env=4 max_batches=15 seed=42 \
       settle.num_steps=480 \
       +settle.validators.support_containment.minimum_resting_heights_m.office_table_background=0.5306 \
       presets=physx render=false --device cpu --viz none

Success means ten rows, with all accepted layouts passing their applicable
checks. A run may need more than ten attempts. Acceptance counts and poses can
vary across machines and backends even with the same seed. ``layouts_per_env``
controls solver pool refills, not the output count. See
:ref:`placement-recording-options` for defaults and the settling duration.

.. _cube-clutter-in-a-container:

Record Cube Clutter in a Bowl
------------------------------

The second maintained scene releases three cubes into a fixed YCB bowl:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/clutter/franka_three_cubes_in_bowl_no_task.yaml \
       output=outputs/clutter/three_cubes_in_bowl.jsonl \
       num_envs=1 min_layouts=1 layouts_per_env=1 max_batches=15 seed=42 \
       settle.num_steps=480 \
       +settle.validators.support_containment.minimum_resting_heights_m.bowl=-0.025 \
       presets=physx render=true --device cpu --viz kit

Here ``bowl`` is the support's runtime scene key. ``-0.025`` is a minimum
resting height in the scaled bowl-local frame, before adding its world position.
It permits settling below the rim. It is not a world-Z coordinate or the rim
height. For another container, measure its own interior floor instead of copying
this value. See :ref:`clutter-support-floor-height`.

The containment check uses the support's bounds and this height. It does not
prove exact containment within curved walls or exclude every possible
penetration. Inspect the contacts as well as the saved validation reports.

.. figure:: ../../../images/clutter/bowl_release.png
   :width: 640px
   :alt: Three cubes released above the rim of the YCB bowl.

   An example release layout above the bowl rim.

.. figure:: ../../../images/clutter/bowl_settled.png
   :width: 640px
   :alt: The same three cubes resting inside the bowl.

   The corresponding accepted layout after settling.

Success produces one row in ``outputs/clutter/three_cubes_in_bowl.jsonl``.
Run the checks in :doc:`qualification`, then inspect it in the viewer:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/clutter/franka_three_cubes_in_bowl_no_task.yaml \
       --placement_layouts outputs/clutter/three_cubes_in_bowl.jsonl \
       --num_envs 1 --device cpu --viz kit

.. _clutter-adapt-environment:

Adapt Your Own Environment
---------------------------

Use the registration and factory path maintained by your environment package.
Register its assets, tasks, and embodiments before resolving them, and start
``SimulationApp`` before simulator-dependent imports. Build the complete
``IsaacLabArenaEnvironment`` description through that factory so its task,
physics callbacks, and runtime configuration are retained. Loading raw YAML is
not a substitute for a factory that also applies these settings.

Prepare the scene using the eligibility checklist above. Resolve ordinary
fixtures to fixed anchors before adding clutter; use
:ref:`ClutterOn settings <clutter-on-relation>` to choose release spread,
clearance, and rotations. Provide measured local heights for non-flat supports.

The following is a runnable built-in equivalent of passing a factory-built
description to the recorder. Save it as ``outputs/clutter/record_python.py``;
the output JSONL path must be unused:

.. code-block:: python

   import argparse

   from isaaclab.app import AppLauncher
   from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

   parser = argparse.ArgumentParser()
   AppLauncher.add_app_launcher_args(parser)
   args = parser.parse_args()

   with SimulationAppContext(args):
       from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
       from isaaclab_arena.offline_placement.recording_config import PlacementRecordingCfg
       from isaaclab_arena.scripts.record_placement_layouts import record_settled_placement_layouts

       source = "isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml"
       arena_env = ArenaEnvGraphSpec.from_yaml(source).to_arena_env()
       # For your package, replace the preceding line with its configured factory call.
       cfg = PlacementRecordingCfg(
           output="outputs/clutter/tools_python.jsonl",
           num_envs=1, min_layouts=1, layouts_per_env=1, max_batches=15,
           presets="physx",
       )
       cfg.settle.num_steps = 480
       cfg.settle.validators["support_containment"] = {
           "minimum_resting_heights_m": {"office_table_background": 0.5306},
       }
       summary = record_settled_placement_layouts(cfg, device=args.device, arena_env=arena_env)
       print(summary.output, summary.accepted, summary.attempted, summary.rejections)
       assert summary.accepted == cfg.min_layouts, "Recording target not reached"

.. code-block:: bash

   python outputs/clutter/record_python.py --device cpu --viz none

The high-level API builds and closes the simulation environment. Supplying
``arena_env`` retains that description's callbacks and settings; ``cfg.presets``
is an explicit backend override. Omit it to keep your environment's preset.
Recording overrides placement sampling settings, including its seed and pool
refill size. Do not reuse those recording seed settings unchanged for replay.
See :ref:`placement-replay-configuration` for the required reset configuration.

If you already own a built environment, use the :ref:`caller-owned recording APIs
<placement-recording-python-api>` and pass the full
``arena_env.get_placement_assets()`` list. These APIs leave the environment open
at its final state. Use ``env.unwrapped`` when accessing Isaac Lab attributes
through a Gym wrapper. Preserve the same geometry, root names, joint resets,
and physics settings when replaying in your environment.
