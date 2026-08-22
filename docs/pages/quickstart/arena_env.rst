First Arena Environment
=======================

Our first environment, ``pick_and_place_maple_table``, is a table-top pick-and-place environment
where a robot picks up an object and places it into a destination container.

.. figure:: ../../images/default_srl_pnp.png
   :width: 100%
   :alt: Default pick_and_place_maple_table environment
   :align: center

   The ``pick_and_place_maple_table`` environment: a DROID robot with a Rubik's cube and bowl on a
   maple table.

Arena builds an environment from three parts that can be changed independently:

* **Scene — the world around the robot.** This includes the room, table, objects, and lighting.
* **Embodiment — the robot.** This defines its cameras, observations, and controls.
* **Task — the job to complete.** Here, the robot must pick up the selected object and place it in
  the selected destination.

For this environment, the three parts are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Part
     - What this example uses
   * - Scene
     - Maple table, Rubik's cube, bowl, and lighting
   * - Embodiment
     - DROID with joint-position control
   * - Task
     - Pick up the cube and place it in the bowl

Keeping these parts separate makes the environment reusable. We can replace the Rubik's cube with
a mustard bottle, choose a different destination or background, or use another compatible robot
without rewriting the task.

The following examples reuse the same environment definition. First, we inspect the reference
scene. Then we replace the objects and change the background. The final visual previews 64 copies
of the scene running in parallel, which you will launch on the next page.


.. _swapping-environment-components:

Run the Environment
-------------------

For this first look, the **Environment Runner** opens one environment in Kit and applies zero
actions, so no policy or model weights are required. It keeps running until you close Kit or press
``Ctrl-C``. While it runs, hold ``Shift`` and left-drag an object to move it and inspect its physical
behavior.

Start or enter the Base Docker container from the repository root:

:docker_run_default:


Run the reference scene
^^^^^^^^^^^^^^^^^^^^^^^

Run the environment with a Rubik's cube, bowl, and home-office background:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab \
     --hdr home_office_robolab

.. figure:: ../../images/default_srl_pnp.png
   :width: 100%
   :alt: Reference DROID pick-and-place environment with a Rubik's cube and bowl
   :align: center

   The reference scene opened in Kit.

Close Kit before starting the next example.


Swap the objects
^^^^^^^^^^^^^^^^

Keep the same environment and task, but replace the pick-up object and destination:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object mustard_bottle_hot3d_robolab \
     --destination_location wooden_bowl_hot3d_robolab \
     --hdr home_office_robolab

.. figure:: ../../images/swap_objects.gif
   :width: 100%
   :alt: Swapping objects in the pick-and-place environment
   :align: center

   The same environment definition and task with different pick-up and destination objects.


Change the background
^^^^^^^^^^^^^^^^^^^^^

Keep the original objects, but select a different background panorama and ambient lighting:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab \
     --hdr billiard_hall_robolab

.. figure:: ../../images/swap_hdr.gif
   :width: 100%
   :alt: Changing the fixed HDR background in the pick-and-place environment
   :align: center

   The same environment definition with a different background panorama.


Preview parallel environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Environment Runner is limited to one interactive environment. On the next page, the
Experiment Runner creates 64 copies of the reference scene that step together on the GPU. This is
what that setup looks like:

.. figure:: ../../images/scale_up.gif
   :width: 100%
   :alt: Running 64 copies of the pick-and-place environment in parallel
   :align: center

   The reference scene scaled from one environment to 64 parallel environments.

.. note::

   The three commands above make explicit choices for each launch. On the next page, those choices
   and the parallel setup become four named Runs in one YAML file. Later, Arena *variations* will
   choose and record values automatically when an environment is built or reset.


How the Environment Is Assembled
--------------------------------

Under the hood, Arena assembles this environment in seven steps:

1. **Retrieve assets.** Arena selects the background, objects, and other scene assets from its
   registries using the environment configuration. A registry is a catalog of assets that Arena
   can use.
2. **Describe spatial relationships.** The table is marked as an anchor, and the objects are
   placed on it. Arena turns these relationships into concrete poses when it builds the
   environment.
3. **Configure lighting.** The configuration sets the light intensity and optional HDR background.
4. **Select the embodiment.** The chosen embodiment provides the robot, cameras, observations, and
   controls.
5. **Compose the scene.** Arena collects the background, lighting, objects, and their spatial
   references into a scene. The robot remains separate from the scene.
6. **Define the task.** The task describes the objective, success and failure conditions, and
   evaluation metrics independently of the selected assets.
7. **Assemble the environment.** Arena combines the scene, embodiment, and task into an environment
   that Isaac Lab can run.

For more detail, see :doc:`Assets <../concepts/scene/concept_assets_design>`,
:doc:`Scenes <../concepts/scene/index>`, :doc:`Embodiments <../concepts/embodiment/index>`,
:doc:`Tasks <../concepts/task/index>`, and
:doc:`Environment Builder <../concepts/environment/env_builder>`.

.. dropdown:: Full source: ``pick_and_place_maple_table_environment.py``
   :animate: fade-in

   .. literalinclude:: ../../../isaaclab_arena_environments/pick_and_place_maple_table_environment.py
      :language: python


Next Steps
----------

Continue to :doc:`arena_experiment` to execute several named Runs from one YAML file.


Using IsaacLab-Arena in Your Own Repository
-------------------------------------------

See :doc:`../arena_in_your_repo/external_installation` for the recommended pattern of consuming
Arena as an unmodified submodule from an external project.
