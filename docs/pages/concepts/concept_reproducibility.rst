Reproducibility
===============

Arena uses two seeds, ``seed`` and ``placement_seed``, so that you can run the same
Experiment again and get the same setup in every episode. ``seed`` keeps the variation
values the same, and ``placement_seed`` keeps the object layouts the same. This page shows
what stays the same from one run to the next, how to control it, and when the physics
simulation can still play out differently.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 46 22 32

   * - What
     - Same in every run?
     - Set with
   * - :doc:`Variation <variations/variations>` values, such as lighting, the background
       image, camera position, object mass, and randomly chosen objects
     - Yes, by default
     - ``seed`` (default ``42``)
   * - Small random offsets that some robots, such as DROID and Franka, add to their
       starting joint positions on every reset
     - Yes, by default
     - ``seed``
   * - Object layouts: where objects are placed
     - Only when you set it
     - ``placement_seed`` (not set by default)
   * - Physics: how objects fall, slide, and collide
     - Only when the whole run repeats
     - See :ref:`reproducibility-physics`

``seed`` and ``placement_seed`` both belong to ``environment_builder`` in an
:doc:`Experiment Definition <concept_arena_experiments>`. Set them under ``shared`` for
every Run, or in a single Run. Scripts that build one environment directly, such as
``environment_runner.py``, take them as ``--seed`` and ``--placement_seed``.

Examples
--------

Same object layouts in every run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``placement_seed``. Without it, objects are placed differently each time you run
the Experiment.

.. code-block:: yaml

   shared:
     environment_builder:
       placement_seed: 7

New variation values
~~~~~~~~~~~~~~~~~~~~

Change ``seed``. Every enabled variation draws new values, and runs with the new seed
match each other again.

.. code-block:: yaml

   shared:
     environment_builder:
       seed: 7

To try another ``seed`` without editing the file, override it on the command line:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config <your_experiment>.yaml \
     shared.environment_builder.seed=7

Several rebuilds
~~~~~~~~~~~~~~~~

Each rebuild of a Run adds one to ``seed``. With the default ``seed`` of 42 and
``num_rebuilds: 3``, the rebuilds use 42, 43, and 44, so each rebuild draws its own
background image. ``placement_seed`` stays the same for every rebuild. Running the
Experiment again gives the same three backgrounds in the same order.

For example, in a Run of the ``pick_and_place_maple_table`` environment:

.. code-block:: yaml

   num_rebuilds: 3
   variations:
     light.hdr_image.enabled: true

Different number of parallel environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``num_envs`` the same between runs that you want to compare, because object
layouts and physics can change with it. To repeat the exact conditions of an earlier
run with a different ``num_envs``, record them and replay them, as described in
:doc:`experiment_record_replay`.

.. _reproducibility-physics:

Physics
-------

``seed`` and ``placement_seed`` decide how each episode is set up. From there, the
physics simulation decides how objects fall, slide, and collide. The physics is
reproducible too, but only when the whole run repeats from the start.

**Physics repeats when you run the same Experiment again** with the same settings,
including ``num_envs``, on the same machine. Every episode in every environment then
plays out exactly as before, as long as the policy sends the same actions.

Three things can still make a rerun differ:

* **Deformable objects.** PhysX only promises repeatable results for rigid objects and
  robots, not for cloth or soft bodies.
* **Physics properties that change during a run**, such as the object mass variation.
  On the GPU, these changes can reorder the simulation's work and slightly change the
  results.
* **Camera images.** Rendering can differ slightly between runs, so a policy that uses
  cameras may not send the same actions. Add ``--deterministic`` to the runner command
  to request repeatable rendering.

**Physics can differ, even from an identical setup,** in these cases:

* **A different number of environments, device, or GPU.** The simulation computes
  slightly differently, and small differences grow as the episode goes on.
* **Two environments in the same run.** Each environment sits at its own place in the
  simulated world. Environment 0 and environment 1 can end slightly differently, even
  when they start from the same layout.
* **An earlier and a later episode in the same run.** Even if episode 1 and episode 5
  start from the same layout and the robot makes the same moves, episode 5 can end
  differently. A reset puts objects back in their starting poses, but the physics engine
  keeps some internal state from earlier episodes. Rerunning the Experiment still
  repeats episode 5 exactly.

The differences are largest when objects touch or are stacked, where a small change can
add up to a visibly different arrangement. Objects that stand apart usually end up in
nearly the same place.

.. note::

   ``placement_seed`` repeats where objects are placed, but objects can still settle a
   little differently in the first moments of an episode. To start every episode with the
   objects already at rest in the same place, record settled layouts once and replay them.
   See :doc:`offline_placement/recording`.

This section describes PhysX, Arena's default physics engine. For background on
simulation determinism, including rendering and the Newton physics engine, see Isaac
Lab's `Reproducibility and Determinism
<https://isaac-sim.github.io/IsaacLab/main/source/how-to/reproducibility.html>`_ guide.
