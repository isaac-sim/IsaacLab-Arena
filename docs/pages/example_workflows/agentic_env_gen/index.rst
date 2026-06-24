Agentic Environment Generation and Policy Evaluation
====================================================

Agentic environment generation creates Arena environments from natural-language
prompts, then reuses the generated environment graph specs for downstream policy
evaluation. This workflow shows how agentically composed environments can be
used by the policy runner, the sequential batch evaluation runner with the
variation system, and policy-specific evaluation flows such as GR00T and PI.

Behind the scenes, this workflow introduces the intent spec, environment graph
spec, and environment graph linking.
For more, see the :doc:`../concepts/concept_overview`.

.. todo:: add concept overview page


Prompt to Environment Graph Spec
--------------------------------

Use the agentic generation runner to resolve a prompt into environment graph
specs:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode resolve \
      --prompt "Droid picks up the mustard bottle from the maple table and places it in the grey bin."

The runner writes files under ``isaaclab_arena_environments/agent_generated/`` by
default:

* ``*_initial.yaml``: the direct output of intent compilation.
* ``*_linked.yaml``: the linked environment graph used by Arena runtime tools.

Pass the linked YAML to policy and evaluation commands.

Prompt to Simulation Environment
--------------------------------

  Use the agentic generation runner to build a simulation environment from prompt-specified environment:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode full \
      --prompt "Droid picks up the mustard bottle from the maple table and places it in the grey bin."

Available Generated Specs
-------------------------

The ``robolab`` subfolder contains example environment graph specs that can be used
directly with policy and evaluation commands:

* ``isaaclab_arena_environments/robolab/bin_mug_marker_bowl_linked.yaml``
* ``isaaclab_arena_environments/robolab/butter_raisin_box_grey_bin_linked.yaml``
* ``isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml``
* ``isaaclab_arena_environments/robolab/bagel_plate_banana_bowl_linked.yaml``


Run a Generated Environment
---------------------------

Generated environments are consumed through ``--env_graph_spec_yaml``:

.. code-block:: bash

   /isaac-sim/python.sh -m isaaclab_arena_examples.policy_runner \
      --viz kit \
      --policy_type zero_action \
      --enable_cameras \
      --num_steps 100 \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml

The same YAML can also be built directly by the generation runner:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena_examples/agentic_environment_generation/environment_generation_runner.py \
      --mode build \
      --linked_env_graph_spec_yaml isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml \
      --headless


Policy Runner with Variations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :doc:`../concepts/variations/variations` for more details on variations.

An Arena environment represented by an environment graph spec YAML can be run
with variations through the policy runner:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type zero_action \
      --enable_cameras \
      isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml \
      light.hdr_image.enabled=true \
      droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true

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

   eval_with_gr00t
   eval_with_openpi

Warnings
--------

This is an active development area, and we are working on the following:

* Support for more complex scene layouts and object placements.
* Support for more complex task specifications.
* Support for interactive environment editing.
* Support in-sim validation for physics and reachability.
* ...
