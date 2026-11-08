Object Control Runner
=====================

The Object Control Runner checks whether an authored environment can satisfy its task predicates
without requiring a robot policy. A JSONL recipe gives pose and joint-position targets. The runner
drives existing scene objects toward those targets with bounded forces instead of teleporting them,
then reports whether the task completed.

Use ``move`` and ``lift`` for object-pose targets, ``joint`` for articulation targets, and
``release`` to stop controlling an object pose. ``wait`` advances simulation, while ``mark``,
``check``, and ``wait_final`` record and validate task progress. ``inspect`` prints the current
state, and ``finish`` ends a live command stream. A ``setup_joint`` record must appear before
runtime commands when an articulation needs an effort-controlled actuator.

The demo always overrides the environment to 120 Hz physics and control (``sim.dt = 1/120``,
``decimation = 1``) before construction. Each ``env.step()`` advances exactly one physics tick,
so movement targets and explicit step budgets use the same clock. Episode timeouts remain in
seconds. Consecutive-step success requirements and tracked events are rescaled to preserve their
original minimum hold durations, rounded up to a 120 Hz tick. Other predicates are evaluated at
the new control rate. Lab records every eighth step for 15 FPS video, including progress overlays.

Results
-------

Viewport recording is enabled by default and requires the Kit visualizer. Each run writes its
artifacts under ``--output``:

* ``validation.json`` contains the final pass/fail result, task state, poses, joints, and events.
* ``demo.mp4`` is the recorded viewport with task-progress annotations.
* ``progress.json`` and boundary images and state files record subtask transitions.
* A failed run writes ``failure_<reason>_...json`` diagnostics and, when available, a viewport PNG.
* ``timings.json`` reports simulator startup and nested wall-clock timers for scene preparation,
  loading, reset, recipe commands, reporting, video postprocessing, and environment cleanup.
  Parent timers include their children; do not sum both when computing total runtime.

Pass ``--no_record --visualizer none`` for a headless run without viewport artifacts.

Open a cabinet
--------------

Run the checked-in cabinet recipe inside the Arena container:

.. code-block:: bash

   isaaclab.sh -p isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml \
     --commands isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2_object_control.jsonl \
     --output artifacts/object_control/open_cabinet \
     --placement_seed 42 \
     --visualizer kit

.. image:: ../../images/kitchen_open_cabinet_object_control.gif
   :alt: Object control opening a cabinet door in the Lightwheel kitchen
   :width: 100%

Recipe:

.. literalinclude:: ../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2_object_control.jsonl
   :language: json

Place bananas in a bowl
-----------------------

The recipe gives both bananas a consistent orientation and separate landing positions, then
releases them above the bowl interior. It runs against the benchmark scene without changing its
task predicates or thresholds.

Run the checked-in banana-placement recipe:

.. code-block:: bash

   isaaclab.sh -p isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml \
     --commands isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2_object_control.jsonl \
     --output artifacts/object_control/place_bananas \
     --placement_seed 42 \
     --visualizer kit

.. image:: ../../images/kitchen_place_bananas_object_control.gif
   :alt: Object control placing two bananas in a bowl in the Lightwheel kitchen
   :width: 100%

Recipe:

.. literalinclude:: ../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2_object_control.jsonl
   :language: json

Live command stream
-------------------

Create a JSONL file containing any required ``setup_joint`` records, then start the runner with
``--follow_commands``. Append newline-terminated runtime commands from another process. The runner
waits for complete lines and exits when it receives ``{"action":"finish"}``. See the
`runner source <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/agentic_tools/object_control_runner.py>`_
for the complete command and output reference.
