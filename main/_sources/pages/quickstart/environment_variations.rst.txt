Exploring Environment Variations
================================

The earlier :ref:`environment examples <swapping-environment-components>` chose one fixed HDR
background for each launch. An Arena variation instead samples a registered environment property
automatically. In this guide, Arena varies the HDR background, light intensity, and wrist-camera
position. Before connecting a trained policy, you can explore these variations with the
``zero_action`` policy. The robot stays still while the environment loads and renders, so no model
weights are required.

The gallery below previews supported lighting, camera, and background effects.

.. figure:: ../../images/lighting_variations_2x2_grid.gif
   :width: 100%
   :alt: Lighting direction, color, temperature, and intensity variations
   :align: center

   Lighting direction, color, temperature, and intensity variations.

.. figure:: ../../images/camera_variations_1x2.gif
   :width: 100%
   :alt: Wrist-camera views with intrinsics and extrinsics variations enabled
   :align: center

   Wrist-camera intrinsics and extrinsics variations.

.. figure:: ../../images/hdr_variations.gif
   :width: 100%
   :alt: The same pick-and-place scene rendered with different HDR backgrounds
   :align: center

   The same task with different HDR backgrounds.


Run the example
---------------

The repository includes a DROID pick-and-place example that enables three variations: the HDR
background, light intensity, and wrist-camera position. Its zero-action policy keeps the robot
still, and five environment rebuilds make the visual changes easy to inspect.

.. dropdown:: Configuration file (``droid_pnp_variations_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml
      :language: yaml

Start or enter the Base Docker container from the repository root:

:docker_run_default:

Then run the example inside the container:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml

The viewport will show the environment being rebuilt with different backgrounds and light
intensities:

.. figure:: ../../images/droid_pnp_variations.gif
   :width: 100%
   :alt: DROID pick-and-place environment rebuilt with different backgrounds and light intensities
   :align: center

   The example at 5x playback speed. The wrist-camera position also changes, but that change
   is not visible from the external viewport.

The background and light intensity are sampled before each environment build. The wrist-camera
position is sampled when the environment resets. Arena calls these *build-time* and *run-time*
variations. See :ref:`build-time-run-time-variations` for details.


Explore other variations
------------------------

Use the same YAML file with ``--list_variations`` to see the variations available for its
environment:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_variations_experiment.yaml \
     --list_variations

See :doc:`../concepts/variations/variations` for the available variations and their
configuration options.

See :doc:`Arena Experiments <../concepts/concept_arena_experiments>` for configuration
rules and local and OSMO execution.


Next steps
----------

Continue to :doc:`running_a_real_policy/index` to replace the zero-action policy with a trained
policy. To measure which variations are associated with success or failure, follow the
:doc:`Evaluation workflow <../example_workflows/analysis/index>`.
