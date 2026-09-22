Scene
=====

A scene is a collection of assets.
It defines the physical environment the robot operates in, and because it
is independent of the robot and task, you can swap objects or backgrounds
without touching the task or embodiment.

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()

   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   scene = Scene(assets=[background, pick_object])

   environment = IsaacLabArenaEnvironment(
       name="my_scene",
       scene=scene,
   )

Assets are loaded from the asset registry by name. An asset can be a
background, a rigid object, or a set of objects (``RigidObjectSet``).
Assets can also carry affordances (e.g. ``Openable``, ``Placeable``) that
describe how they can be interacted with, which is what allows tasks to
work with any compatible object.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_assets_design
   concept_affordances_design
