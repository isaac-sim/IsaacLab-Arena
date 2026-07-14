Evaluation Types
================

The **Experiment Runner** is the typed entry point for policy evaluation:

``isaaclab_arena/evaluation/experiment_runner.py``

The same runner configuration and Arena Experiment YAML select either of two
execution routes. Pass ``--local`` to execute the Experiment in the current
Arena container. Omit ``--local`` to submit the complete Experiment as one OSMO
workflow. A Run is not submitted as an independent workflow: an Experiment with
many Runs stays one Experiment on both routes.

The older ``policy_runner.py`` interface and the JSON/``argparse.Namespace``
evaluation path are deprecated compatibility interfaces. They remain useful
while existing examples migrate, but new evaluation workflows should use typed
YAML and the Experiment Runner.

For the types used inside an Experiment, see :doc:`Policy Design <index>`,
:doc:`Environment Design <../concept_overview>`, and
:doc:`Metrics Design <../task/concept_metrics_design>`.

.. _experiment-runner:
.. _sequential-batch-experiment-runner:

Experiment Runner
-----------------------

The runner keeps three concerns separate:

.. list-table:: Configuration layers
   :header-rows: 1
   :widths: 24 42 34

   * - Configuration
     - Owns
     - Used by
   * - Experiment Runner YAML
     - Experiment path, output, video, failure handling, and report settings
     - Local and OSMO routes
   * - Arena Experiment YAML
     - The ordered Runs and each Run's typed environment, policy, and rollout limit
     - Local and OSMO routes
   * - OSMO workflow YAML
     - Images, compute resources, timeouts, output storage, and policy-server infrastructure
     - OSMO route only

The scientific Experiment YAML is therefore independent of where it runs. OSMO
settings do not leak into the Experiment, and local Isaac Sim launcher settings
do not become part of it.

Runner configuration
^^^^^^^^^^^^^^^^^^^^

The YAML passed to ``--config`` is parsed as an
``ExperimentRunnerCfg``. Its required ``experiment_config`` value may be
absolute or relative to the runner configuration file:

.. code-block:: yaml

   experiment_config: experiment.yaml
   output_base_dir: ./output
   record_viewport_video: false
   record_camera_video: false
   continue_on_error: false
   serve_evaluation_report: false
   evaluation_report_port: 8000

Runner settings describe how to execute and report the Experiment. They do not
duplicate environment or policy fields from its Runs.

Experiment configuration
^^^^^^^^^^^^^^^^^^^^^^^^

An Arena Experiment is an ordered ``runs`` collection. Every Run selects a
registered, typed environment and policy and defines a rollout limit. For
example:

.. code-block:: yaml

   runs:
   - name: openpi_maple_table
     environment:
       type: pick_and_place_maple_table
       enable_cameras: true
       embodiment: droid_abs_joint_pos
       pick_up_object: rubiks_cube_hot3d_robolab
       destination_location: bowl_ycb_robolab
       hdr: home_office_robolab
     policy:
       type: pi0_remote
       policy_variant: pi05
     rollout_limit:
       num_episodes: 1

Typed loading rejects unknown fields instead of silently forwarding arbitrary
command-line values. Policies and environments receive their registered config
types directly; they do not need to reconstruct their configuration from an
``argparse.Namespace``.

Run locally
^^^^^^^^^^^

Use ``--local`` when the current Arena container should own Isaac Sim and
execute every Run:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --config path/to/runner.yaml \
     --local \
     --enable_cameras

Arguments owned by Isaac Lab's ``AppLauncher``, such as ``--headless``,
``--device``, and currently ``--enable_cameras``, are accepted only on the local
route. A local remote-policy Run expects its policy server to be reachable
already.

Submit to OSMO
^^^^^^^^^^^^^^

Omitting ``--local`` builds and submits one OSMO workflow for the complete
Experiment, then finishes after submission:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --config path/to/runner.yaml

The bundled infrastructure defaults come from
``osmo/config/arena_experiment_workflow.yaml``. Select another typed OSMO
configuration without changing the Experiment:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --config path/to/runner.yaml \
     --osmo-config path/to/osmo.yaml

The workflow stages the source Experiment YAML and derives the remote runner
configuration for the container. When the Experiment contains OpenPI Runs, the
workflow also starts one shared OpenPI server and injects its endpoint into
those Runs at submission time; the source Experiment YAML remains unchanged.

Hydra overrides
^^^^^^^^^^^^^^^

Trailing Hydra overrides apply to the Arena Experiment after its YAML values.
They work on both routes and are appended after overrides stored in the runner
configuration:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --config path/to/runner.yaml \
     --local \
     --enable_cameras \
     runs.openpi_maple_table.rollout_limit.num_episodes=10

Use the Run name in the override path. Keep repeatable values in YAML and use
trailing overrides for deliberate, invocation-specific changes.

Route summary
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Route
     - Selection
     - Result
   * - Local
     - Add ``--local``
     - Execute all Runs in the current container
   * - OSMO
     - Omit ``--local``
     - Submit all Runs as one workflow and return after submission

The Experiment Runner does not currently provide the deprecated policy
runner's ``torchrun``-based distributed mode. Configure OSMO resources in the
OSMO workflow YAML, or divide work into intentionally separate Experiments.

Legacy compatibility path
-------------------------

Invoking ``experiment_runner.py`` without ``--config`` selects the
deprecated interface. That path accepts legacy JSON job files, reconstructs
configuration through ``argparse.Namespace``, and preserves older runner flags
while callers migrate. ``policy_runner.py`` is deprecated for the same reason.

Do not use the compatibility path as the basis for new configuration schemas.
In particular, do not add new ``--eval_jobs_config`` fields or translate typed
YAML back into CLI argument dictionaries. Define fields on the appropriate
typed environment, policy, Experiment, runner, or OSMO configuration instead.
