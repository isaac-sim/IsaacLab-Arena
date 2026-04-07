Assets
======

Assets are the objects and backgrounds that populate a scene.
All assets are registered in the asset registry and loaded by name.

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   cracker_box = asset_registry.get_asset_by_name("cracker_box")()

Discovering assets
------------------

To find what is available, assets can be queried by tag:

.. code-block:: python

   # Get all objects tagged as pickable
   objects = asset_registry.get_assets_by_tag("pickable")

   # Pick one at random
   obj = asset_registry.get_random_asset_by_tag("pickable")()

Common tags include ``"pickable"``, ``"background"``, and ``"openable"``.

Object types
------------

Every asset has an object type that determines how it is simulated:

- **RIGID** — a single rigid body (boxes, bottles, tools, furniture).
- **ARTICULATION** — a multi-body object with joints (robots, doors, drawers, appliances).
- **BASE** — no physics; used for static backgrounds and markers.

You generally don't need to set this yourself — assets in the registry already
have the correct type defined.

Object references
-----------------

A background asset like a kitchen is a single USD file containing many prims:
countertops, shelves, drawers, and so on. To use one of these internal prims
as a destination or interaction target (e.g. "place the object on the counter"),
you use an ``ObjectReference``.

.. code-block:: python

   kitchen = asset_registry.get_asset_by_name("kitchen")()

   counter = ObjectReference(
       name="kitchen_counter",
       prim_path="{ENV_REGEX_NS}/kitchen/counter_right_main_group/top_geometry",
       parent_asset=kitchen,
   )

   task = PickAndPlaceTask(
       pick_up_object=cracker_box,
       destination_location=counter,
       background_scene=kitchen,
   )

The ``parent_asset`` tells the environment which spawned USD the prim path belongs to.
The prim path uses ``{ENV_REGEX_NS}`` so it resolves correctly across parallel environments.
