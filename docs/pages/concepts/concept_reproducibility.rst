Reproducibility
===============

Arena uses two seeds, ``seed`` and ``placement_seed``, so that you can launch the same
Experiment again and get the same setup in every episode. ``seed`` keeps the variation
values the same, and ``placement_seed`` keeps the object layouts the same. This page shows
what stays the same from one launch to the next, how to control it, and when the physics
simulation can still play out differently.

A *launch* is one start of the Experiment Runner. A *Run* is one entry in an
:doc:`Experiment Definition <concept_arena_experiments>`, and one launch executes all
Runs of the Experiment.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 46 22 32

   * - What
     - Same in every launch?
     - Set with
   * - :doc:`Variation <variations/variations>` values, such as lighting, the background
       image, camera position, object mass, and randomly chosen objects
     - Yes, by default
     - ``seed``
   * - Small random offsets that some robots, such as DROID and Franka, add to their
       starting joint positions on every reset
     - Yes, by default
     - ``seed``
   * - Object layouts: where objects are placed
     - Only when you set it
     - ``placement_seed``
   * - Physics: how objects fall, slide, and collide
     - Only under some conditions
     - See :ref:`reproducibility-physics`

``seed`` and ``placement_seed`` both belong to ``environment_builder`` in an
:doc:`Experiment Definition <concept_arena_experiments>`. Set them under ``shared`` for
every Run, or in a single Run.

Examples
--------

Same object layouts in every launch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``placement_seed``. Without it, objects are placed differently each time you launch
the Experiment.

.. code-block:: yaml

   shared:
     environment_builder:
       placement_seed: 7

New variation values
~~~~~~~~~~~~~~~~~~~~

Change ``seed``. Every enabled variation that draws at random gets new values, and
launches with the new seed match each other again. Asset selection takes objects in list order by
default and ignores ``seed``; see :ref:`asset selection <asset-selection-variation>`.

.. code-block:: yaml

   shared:
     environment_builder:
       seed: 7

Once ``seed`` is declared under ``shared.environment_builder`` as above, you can try
other values on the command line without editing the file again:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config <your_experiment>.yaml \
     shared.environment_builder.seed=7

Several rebuilds
~~~~~~~~~~~~~~~~

Each rebuild of a Run adds one to ``seed``. With ``seed: 42`` and ``num_rebuilds: 3``,
the rebuilds use 42, 43, and 44, so each rebuild draws its own background image.
``placement_seed`` stays the same for every rebuild. Launching the Experiment again gives
the same three backgrounds in the same order.

For example, in a Run of the ``pick_and_place_maple_table`` environment:

.. code-block:: yaml

   num_rebuilds: 3
   variations:
     light.hdr_image.enabled: true

Different number of parallel environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep ``num_envs`` the same between launches that you want to compare, because object
layouts and physics can change with it. To reuse recorded variation values and object
layouts with a different ``num_envs``, record them and replay them, as described in
:doc:`experiment_record_replay`. Other random reset events, such as the robots' joint
offsets, are not recorded, so the robot's start and the physics can still differ.

.. _reproducibility-physics:

Physics
-------

``seed`` and ``placement_seed`` decide how each episode is set up. From there, the
physics simulation decides how objects fall, slide, and collide.

**Physics repeats when you launch the same Experiment again**: every episode in every
environment plays out exactly as in the previous launch. This holds when all of the
following are true:

* ``placement_seed`` is set, and every other setting, including ``num_envs``, is the
  same.
* The launch runs on the same machine, with the same device and GPU. Other hardware
  computes slightly differently, and small differences grow as the episode goes on.
* The scene has only rigid objects and robots. PhysX does not promise repeatable results
  for cloth or soft bodies.
* No variation changes physics properties while the simulation runs, such as the object
  mass variation. On the GPU, these changes can reorder the simulation's work and
  slightly change the results.
* The policy sends the same actions. Camera images can differ slightly between launches,
  so a policy that uses cameras may not. Add ``--deterministic`` to the runner command to
  request repeatable rendering.

**Within one launch, physics does not repeat**, even from an identical setup:

* **Two environments.** Each environment sits at its own place in the simulated world.
  Environment 0 and environment 1 can end slightly differently, even when they start
  from the same layout.
* **An earlier and a later episode.** Even if episode 1 and episode 5 start from the
  same layout and the robot makes the same moves, episode 5 can end differently. A reset
  puts objects back in their starting poses, but the physics engine keeps some internal
  state from earlier episodes. Launching the Experiment again still repeats episode 5
  exactly.

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
