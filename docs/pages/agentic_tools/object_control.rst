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

Results
-------

Viewport recording is enabled by default and requires the Kit visualizer. Each run writes its
artifacts under ``--output``:

* ``validation.json`` contains the final pass/fail result, task state, poses, joints, and events.
* ``demo.mp4`` is the recorded viewport with task-progress annotations.
* ``progress.json`` and boundary images and state files record subtask transitions.
* A failed run writes ``failure_<reason>_...json`` diagnostics and, when available, a viewport PNG.

Pass ``--no_record --visualizer none`` for a headless run without viewport artifacts.

Open a cabinet
--------------

Run the checked-in cabinet recipe inside the Arena container:

.. code-block:: bash

   isaaclab.sh -p isaaclab_arena_examples/agentic_environment_generation/object_control_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml \
     --commands isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2_object_control.jsonl \
     --output artifacts/object_control/open_cabinet \
     --visualizer kit

.. image:: ../../images/kitchen_open_cabinet_object_control.gif
   :alt: Object control opening a cabinet door in the Lightwheel kitchen
   :width: 100%

Recipe:

.. literalinclude:: ../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2_object_control.jsonl
   :language: json

Place bananas in a bowl
-----------------------

Run the checked-in banana-placement recipe:

.. code-block:: bash

   isaaclab.sh -p isaaclab_arena_examples/agentic_environment_generation/object_control_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml \
     --commands isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2_object_control.jsonl \
     --output artifacts/object_control/place_bananas \
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
`runner source <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_examples/agentic_environment_generation/object_control_runner.py>`_
for the complete command and output reference.
