Record and Replay Clutter Layouts
=================================

``ClutterOn`` places objects above a support so physics can settle them into a
reusable layout. Use ``record_placement_layouts.py`` to save accepted root poses
to JSONL, then replay those poses without solving or settling again.

Complete :doc:`../../quickstart/installation` and run the commands below from the
repository root. Recording can run headless: replace ``render=true --viz kit``
with ``render=false --viz none``. The interactive replay commands below require
a display; use the Experiment workflow in :doc:`recording` for headless replay.

Tool Clutter on a Table
-----------------------

The ``franka_three_hammers_and_clamp_no_task`` environment drops three hammers
and a clamp onto a fixed table.

Record
~~~~~~

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       output=outputs/clutter/tools_on_table.jsonl \
       num_envs=4 min_layouts=10 layouts_per_env=4 max_batches=15 seed=42 \
       settle.num_steps=480 \
       +settle.validators.support_containment.minimum_resting_heights_m.office_table_background=0.5306 \
       render=true --device cpu --viz kit

The table has a beveled top, so the command supplies its local-Z top height.

.. figure:: ../../../images/clutter/release.png
   :width: 640px
   :alt: Three hammers and a clamp suspended above a table before settling.

   Release poses selected by the placement solver.

.. figure:: ../../../images/clutter/settled.png
   :width: 640px
   :alt: Three hammers and a clamp resting on a table after settling.

   The recorded root poses after physics settling.

Replay
~~~~~~

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       --placement_layouts outputs/clutter/tools_on_table.jsonl \
       --num_envs 1 --device cpu --viz kit

Cube Clutter in a Container
---------------------------

The ``franka_three_cubes_in_bowl_no_task`` environment drops three cubes into a
fixed YCB bowl.

Record
~~~~~~

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/clutter/franka_three_cubes_in_bowl_no_task.yaml \
       output=outputs/clutter/three_cubes_in_bowl.jsonl \
       num_envs=1 min_layouts=1 layouts_per_env=5 max_batches=5 seed=42 \
       settle.num_steps=480 \
       +settle.validators.support_containment.minimum_resting_heights_m.bowl=-0.025 \
       render=true --device cpu --viz kit

The minimum resting height is the bowl floor in the bowl's local frame. It lets
the cubes settle below the rim while still rejecting objects that fall through
the bowl.

.. figure:: ../../../images/clutter/bowl_release.png
   :width: 640px
   :alt: Three cubes above a bowl before settling.

   Release poses above the bowl rim.

.. figure:: ../../../images/clutter/bowl_settled.png
   :width: 640px
   :alt: Three cubes resting inside a bowl after settling.

   The recorded root poses after physics settling.

Replay
~~~~~~

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/clutter/franka_three_cubes_in_bowl_no_task.yaml \
       --placement_layouts outputs/clutter/three_cubes_in_bowl.jsonl \
       --num_envs 1 --device cpu --viz kit

Recording and Replay Notes
--------------------------

Choose a new output path for each recording; existing JSONL files are not
overwritten. Each line stores one accepted layout. If the batch budget ends
early, any accepted layouts are still written; no file is created when none pass.

Both examples use ``NoTask``, so interactive replay loads the first saved layout
and does not trigger episode resets. Close the viewer to exit. For policy
evaluation replay, use the Experiment workflow in :doc:`recording`.

See :doc:`../object_placement/validation` for clutter acceptance checks,
container-height settings and rejection guidance. See
:doc:`../object_placement/relations` for the JSONL format and layout selection
across resets and parallel environments.
