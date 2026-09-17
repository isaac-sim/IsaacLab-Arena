Clutter Placement Example
=========================

``clutter_scene.yaml`` places four cubes above a fixed table using ``ClutterOn``.
The example uses PhysX. Generate settled layouts with Arena's offline script:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/generate_clutter_scene.py \
       --env_spec isaaclab_arena_examples/relations/clutter/clutter_scene.yaml \
       --output outputs/clutter/placements.yaml --num_envs 4 --num_layouts 100 \
       --seed 42 --viz none

Generation controls, replay behavior, and validation limits are documented in
``docs/pages/concepts/object_placement/clutter_placement.rst``.
