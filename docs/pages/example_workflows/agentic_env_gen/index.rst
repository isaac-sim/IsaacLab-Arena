Agentic Environment Generation and Policy Evaluation
====================================================

Agentic environment generation creates Arena environments from natural-language
prompts, then reuses the generated environment graph specs for downstream policy
evaluation. This workflow shows how agentically composed environments can be
used by the policy runner, the sequential batch evaluation runner with the
variation system, and policy-specific evaluation flows such as GR00T and PI.

Behind the scenes, this workflow introduces the intent spec, environment graph
spec, and environment graph linking.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

.. todo:: add concept overview page


Prompt to Environment Graph Spec
--------------------------------

Use the agentic generation runner to resolve a prompt into environment graph
specs:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode resolve \
      --prompt "Droid picks up the mustard bottle from the maple table and places it in the grey bin."

The runner writes files under ``isaaclab_arena_environments/agent_generated/`` by
default:

* ``*_initial.yaml``: the direct output of intent compilation.
* ``*_linked.yaml``: the linked environment graph used by Arena runtime tools.

Pass the linked YAML to policy and evaluation commands.

Prompt to Simulation Environment
--------------------------------

Use the agentic generation runner to build a simulation environment from a
prompt-specified environment:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode full \
      --prompt "Droid picks up the mustard bottle from the maple table and places it in the grey bin."

Interactive GUI Runner
----------------------

As an alternative to the CLI runner, use the GUI runner to interactively
generate, edit, and visualize the prompt-specified environment in a web browser:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/gui_runner.py

.. note::

   Agent-generated specs may have missing or incorrect fields. We recommend
   using the interactive GUI to manually fix and validate each spec before using
   it for full evaluation.

   For example:

   * ``isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml``
     manually adds a ``rotate_around_solution`` relation to set the raisin box
     in a standup position.
   * ``isaaclab_arena_environments/robolab/two_bin_linked.yaml`` manually edits
     the ``next_to`` relation ``side`` param to set the correct left/right
     positioning in robot coordinates.

See :doc:`gui_runner` for the full UI walkthrough.

Available Generated Specs
-------------------------

The ``isaaclab_arena_environments/robolab`` subfolder contains Arena environment graph specs generated
from RoboLab tasks. See :doc:`../robolab_task_catalog` for the list of RoboLab tasks
currently supported in Arena.

Run a Generated Environment
---------------------------

Generated environments are consumed through ``--env_graph_spec_yaml``:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type zero_action \
      --enable_cameras \
      --num_steps 100 \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml

The same YAML can also be built directly by the generation runner:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode build \
      --linked_env_graph_spec_yaml isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml \
      --headless


Policy Runner with Variations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An Arena environment represented by an environment graph spec YAML can be run
with variations through the policy runner:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type zero_action \
      --enable_cameras \
      isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml \
      light.hdr_image.enabled=true \
      droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true

.. figure:: ../../../images/agentic_env_gen_policy.gif
   :alt: Agentic environment generation with PI policy and HDR variation sensitivity analysis

   Agentically generated environments can be evaluated with policy runners and
   variation sweeps, such as changing the background HDR image to probe policy
   sensitivity.

Sequential Batch Evaluation Runner with Variations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation jobs can also point their environment source at a linked graph YAML
with variations, instead of a registered example-environment name:

.. code-block:: json

   {
       "name": "agentic_env_eval",
       "arena_env_args": {
           "environment": "isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml",
           "enable_cameras": true
       },
       "num_steps": 100,
       "num_rebuilds": 1,
       "policy_type": "zero_action",
       "policy_config_dict": {}
   }

Evaluation Policies Workflow Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the steps below to complete the workflow:

- :doc:`eval_with_gr00t`
- :doc:`eval_with_openpi`


.. toctree::
   :maxdepth: 1
   :hidden:

   gui_runner
   eval_with_gr00t
   eval_with_openpi

Warnings
--------

.. note::

   Agentic environment generation is experimental and changing quickly. The
   current prompt formats, generated spec structure, GUI behavior, and policy
   evaluation integrations may change across releases.

   We are actively working on:

   * Support for more complex scene layouts and object placements.
   * Support for more complex task specifications.
   * Support in-sim validation for physics and reachability.
   * ...
