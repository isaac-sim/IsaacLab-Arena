Agentic Environment Generation
==============================

Motivation
----------

Building evaluation environments manually does not scale. Traditionally,
creating an Arena evaluation environment required users to write Python that
selects assets, defines initial-state distributions and variations, and
configures the task. Users then had to run the simulation and inspect whether
the initial conditions were physically stable and feasible for the selected
robot and task.

.. figure:: ../../../images/agentic_env_gen_user_journy.png
   :width: 100%
   :alt: Agentic environment generation pipeline
   :align: center

   The generated ArenaEnvGraphSpec is the boundary between generation and simulation.
   It can be saved, reviewed, and edited before an Arena environment is built.

The pipeline turns a task described in natural language into an editable
:doc:`ArenaEnvGraphSpec <../environment/environment_definition>`. It differs
from one-off agentic scene-generation workflows in three ways:

* **Arena-native output.** The pipeline generates modular Arena scene and task
  descriptions rather than a one-off simulator scene. The resulting
  environment supports framework features such as
  :doc:`parallel evaluation <../policy/concept_evaluation_types>` and
  :doc:`variations for sensitivity analysis <../variations/variations>`.
* **Task-oriented generation.** Validation checks the logical consistency of
  subtasks and relations. Arena's :doc:`relation solver <../object_placement/solver>`
  then solves object and robot placement from spatial and task constraints, and
  its :doc:`placement validator <../object_placement/validation>` applies configured
  physics-stability and robot-reachability checks.
* **Distributions rather than a single scene.** Arena's relation solver can
  produce multiple initial-state realizations that satisfy the task
  constraints. During policy evaluation it samples diverse valid layouts
  across rollouts, supporting statistically meaningful evaluation rather than
  relying on one hand-authored arrangement.

.. figure:: ../../../images/agentic_environment_generation/tabletop_agentic_env_banana_bagel_plate.png
   :width: 100%
   :alt: Parallel initial-state realizations generated from one environment specification
   :align: center

   One agent-generated
   `YAML specification <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml>`_
   resolves into the many initial-state realizations shown here for the prompt:
   “Droid picks up the banana from the maple table and places it on the plate.
   There are two bagels and one bowl on the table.”

Non-goals
---------

Arena's environment generation agent explicitly does not provide the following:

* **SimReady asset generation.** Generating physically accurate, Isaac
  Sim-ready asset USDs is outside this project's scope. Arena assumes users
  have access to public or private asset databases from which they can build
  benchmark asset libraries; for example, refer to NVIDIA's
  `SimReady project <https://docs.omniverse.nvidia.com/simready/latest/index.html>`_.
* **Background generation.** Arena places objects in existing background USD
  assets; it does not generate realistic rooms or other background USDs.
  Separate Replicator workflows cover that problem.
* **Motion generation.** Demonstration and motion generation belong to the
  Isaac Lab Mimic Next workstream rather than environment generation.

.. toctree::
   :maxdepth: 1

   system_overview
   model_selection
   gui_runner
   cli_runner

.. note::

   Agentic environment generation is experimental. Generated specs should be
   reviewed and validated before they are used for policy evaluation.
