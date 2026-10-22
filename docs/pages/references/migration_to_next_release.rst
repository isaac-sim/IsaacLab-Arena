Migrate from v0.3.1 to the next release
=======================================

This guide tracks changes on ``main`` after the ``release/0.3.1`` branch that require
action in downstream projects. It will be assigned a version during release preparation.

Quick access
------------

.. list-table::
   :header-rows: 1

   * - If your v0.3.1 project...
     - Go to
   * - Configures task success directly
     - :ref:`Task success configuration <migration-next-task-success>`
   * - Imports the environment-generation inference backend
     - :ref:`Inference backend imports <migration-next-inference>`
   * - Uses placement layouts, placement replay, or placement solver helpers
     - :ref:`Relation placement and replay <migration-next-placement>`

.. _migration-next-task-success:

Task success configuration
--------------------------

Task progress is now owned by the Arena environment rather than by
``TaskSuccessTerm``. ``TaskSuccessTerm`` and
``ProgressTrackingRecorderManagerCfg`` were removed.

Replace direct success-term configuration:

.. code-block:: python

   from isaaclab.managers import TerminationTermCfg
   from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm

   env_cfg.terminations.success = TerminationTermCfg(
       func=TaskSuccessTerm,
       params={"success_criteria": criteria},
   )

with:

.. code-block:: python

   from isaaclab.managers import TerminationTermCfg
   from isaaclab_arena.progress_tracking.task_progress_cfg import TaskProgressCfg
   from isaaclab_arena.tasks.terminations import task_success_from_progress

   env_cfg.task_progress = TaskProgressCfg(success_criteria=criteria)
   env_cfg.terminations.success = TerminationTermCfg(func=task_success_from_progress)

``ArenaEnvBuilder`` performs this configuration automatically for Arena tasks.

.. _migration-next-inference:

Inference backend imports
-------------------------

The inference backend moved out of the environment-generation package.

.. code-block:: python

   # Before
   from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend

   # After
   from isaaclab_arena.inference.backend import InferenceBackend

Move imports of ``InferenceEndpoint``, ``StructuredOutputRequest``, and the endpoint
helpers in the same way. The internal endpoint's default model also changed; pass an
explicit model when model identity must remain stable.

.. _migration-next-placement:

Relation placement and replay
-----------------------------

``PlacementLayouts`` was replaced by a list of ``PlacementSample`` values, and
``solve_and_apply_relation_placement`` was replaced by
``prepare_live_relation_placement``. The prepare function returns a
``PlacementSampler`` rather than an Isaac Lab event configuration. Code that creates
the reset event directly must pass that sampler to ``make_placement_event``.

Recorded placement is now part of variation replay. Replace
``ArenaEnvBuilderCfg.placement_layouts_path`` and
``IsaacLabArenaEnvironment(..., placement_layouts=...)`` with:

.. code-block:: python

   ArenaEnvBuilderCfg(recorded_variation_samples_path="path/to/episodes.jsonl")

Placement JSONL rows must also move placement out of
``variations["scene.relation_placement"]`` and into the top-level ``placement`` field:

.. code-block:: json

   {
     "placement": {
       "layout_id": "0",
       "poses": {
         "object": {
           "position_xyz": [0.0, 0.0, 0.5],
           "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
         }
       }
     },
     "variations": {}
   }

The legacy JSONL shape is rejected. Use ``PlacementSample`` and the serialization
helpers in ``isaaclab_arena.relations.placement_sampler`` for custom tooling.
