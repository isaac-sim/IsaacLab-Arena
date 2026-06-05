Listing Hydra-configurable Variations
=====================================

The :doc:`exploring_variations` experiments use **argparse** flags such as ``--hdr`` and
``--pick_up_object`` to swap scene assets. Separately, some assets and embodiments expose
**Hydra-style variation overrides** (dotted paths like ``light.hdr_image.enabled=true``) for
build-time and run-time randomization.

To see which variation keys exist for a given environment, pass ``--list-variations`` to the
policy runner after the usual environment subcommand and flags (uses the same SimulationApp
startup as a normal run, then prints the catalog and exits before rollout):

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --list-variations \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab

You can include Hydra override tokens on the same command line; the printed defaults reflect them.
