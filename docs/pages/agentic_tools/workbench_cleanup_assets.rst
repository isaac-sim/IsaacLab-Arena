Workbench Cleanup Demos
=======================

These object-control recipes exercise the four atomic workbench tasks and their sequential
composition. They use the registered derived assets referenced by
``isaaclab_arena_environments/workbench_cleanup/scene_tool_bin_scope.yaml``.
The DROID robot remains idle: bounded forces and joint efforts move the scene objects directly.
These demonstrations validate the authored tasks, rather than robot-policy performance.

Replay and record
-----------------

Run from the repository root inside the clone's Arena container. The registered
``workbench_cleanup_*`` assets download their pinned SimReady packages and prepare
them automatically on first use. Subsequent runs reuse the derived asset cache; no
preparation script or local USD mount is required. The recording script requires
``ffmpeg`` and a Kit viewport.

.. code-block:: bash

   isaaclab.sh -p isaaclab_arena_environments/workbench_cleanup/scripts/run_object_control_demos.py \
     --output-dir /eval/workbench/object_control_demos

This invokes :doc:`object_control` once per task with simulation and placement seeds of 42.
Use a new output directory for each run. Pass ``--tasks close_drawer`` (or another task name)
to record a subset. Pass ``--convert-only`` to regenerate GIFs from existing successful recordings,
or ``--gif-dir /eval/workbench/demo_gifs`` to keep the GIFs outside the documentation tree.

Each task directory retains the scene and recipe used (``environment.yaml`` and ``commands.jsonl``),
``run.log``, ``validation.json``, ``demo.mp4``, ``timings.json``, progress records, and boundary snapshots.
For the button and full cleanup demos, the script changes only the viewer pose to show the
oscilloscope screen. A failed recipe stops the script and preserves the
runner's failure diagnostics. GIF conversion requires ``success: [true]`` in the validation report.
The GIFs below play at twice simulation speed, at 10 FPS and 800 pixels wide.

The JSONL recipes live under
``isaaclab_arena_environments/workbench_cleanup/object_control/``. To run one directly:

.. code-block:: bash

   isaaclab.sh -p isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_bin_shelf.yaml \
     --commands isaaclab_arena_environments/workbench_cleanup/object_control/bin_shelf.jsonl \
     --output /eval/workbench/bin_shelf_demo \
     --seed 42 --placement_seed 42 --visualizer kit

The recipes retain the task predicates and thresholds. The runner uses its standard 120 Hz
control override and preserves consecutive-step hold durations. ``setup_joint`` replaces the
controlled joint's drive with the recipe's effort controller; pose control is released before
checking whether the white bin has settled on the shelf. Joint targets remain held through
the final check. These recipes demonstrate a single fixed placement seed, not a success-rate
evaluation over randomized layouts.

Close the drawer
----------------

The top-left drawer starts open with two tools inside. The recipe closes the articulated drawer
and checks the existing ``CloseDoorTask`` predicate.

.. image:: ../../images/workbench_cleanup_assets/close_drawer.gif
   :alt: Object control closing the top-left workbench drawer
   :width: 100%

.. literalinclude:: ../../../isaaclab_arena_environments/workbench_cleanup/object_control/close_drawer.jsonl
   :language: json

Close the toolcase
------------------

The socket case lid starts open. The recipe closes its hinge and checks ``CloseDoorTask``.

.. image:: ../../images/workbench_cleanup_assets/close_toolcase.gif
   :alt: Object control closing the socket toolcase lid
   :width: 100%

.. literalinclude:: ../../../isaaclab_arena_environments/workbench_cleanup/object_control/close_toolcase.jsonl
   :language: json

Place the bin on the shelf
--------------------------

The bin moves out in front of the workbench, rises above the shelf, and lowers into the empty bay
between the blue containers. The recipe releases it, then waits for the normal contact, support,
and settling checks to pass.

.. image:: ../../images/workbench_cleanup_assets/bin_shelf.gif
   :alt: Object control lifting and placing the white bin on the overhead shelf
   :width: 100%

.. literalinclude:: ../../../isaaclab_arena_environments/workbench_cleanup/object_control/bin_shelf.jsonl
   :language: json

Turn off the oscilloscope
-------------------------

The recipe depresses the power-button joint and checks ``PressButtonTask``. The scene's existing
joint-triggered material event switches the screen off.

.. image:: ../../images/workbench_cleanup_assets/turn_off_scope.gif
   :alt: Object control pressing the oscilloscope power button
   :width: 100%

.. literalinclude:: ../../../isaaclab_arena_environments/workbench_cleanup/object_control/turn_off_scope.jsonl
   :language: json

Full cleanup
------------

``full_cleanup.jsonl`` performs the four operations in order in a single environment episode:
close the drawer, close the toolcase, place the bin, and press the oscilloscope button.
It checks every subtask and the final sequential-task success condition.

.. image:: ../../images/workbench_cleanup_assets/full_cleanup.gif
   :alt: Object control completing all four workbench cleanup subtasks in one episode
   :width: 100%

.. literalinclude:: ../../../isaaclab_arena_environments/workbench_cleanup/object_control/full_cleanup.jsonl
   :language: json
