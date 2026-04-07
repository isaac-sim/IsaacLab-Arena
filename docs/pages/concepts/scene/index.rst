Scene
=====

A scene is composed of a collection of 3D assets that define the simulated environment
into which our robot will operate.
Defining a scene is easy. You simply pass a list of assets that should be added
to the environment. An asset can be a background or an object.
The code example below shows constructing a simple scene

.. code-block:: python

   # Asset creation and positioning
   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()

   # Position the object on the benchtop.
   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Scene composition
   scene = Scene(assets=[background, pick_object])

   # Environment integration
   environment = IsaacLabArenaEnvironment(
       name="my_scene",
       scene=scene,
   )

Walkthrough
------------

Let's walk through the code example above step-by-step.

First, we load some objects from the asset registry

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()

We then position the object on the benchtop.

.. code-block:: python

   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

We create the scene by composing the background and the object(s).

.. code-block:: python

   scene = Scene(assets=[background, pick_object])

Finally we create a simple Isaac Lab Arena environment.

.. code-block:: python

   environment = IsaacLabArenaEnvironment(
       name="my_scene",
       scene=scene,
   )

Simple as that!

More details
------------

The rest of this section will describe further details of the scenes component.

.. toctree::
   :maxdepth: 1

   concept_assets_design
   concept_affordances_design
