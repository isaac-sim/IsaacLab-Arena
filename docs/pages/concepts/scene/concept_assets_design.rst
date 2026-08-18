Assets
======

Assets are the objects and backgrounds that make up a scene.
Arena ships with a set of assets ready to use by name,
and new assets can be added by registering them in the asset library.

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   cracker_box = asset_registry.get_asset_by_name("cracker_box")()

Registering a new asset
-----------------------

To add a new object, subclass ``LibraryObject``, provide the USD path and object type,
and decorate it with ``@register_asset``:

.. code-block:: python

   @register_asset
   class MyObject(LibraryObject):
       name = "my_object"
       tags = ["object", "graspable"]
       usd_path = "path/to/my_object.usd"
       object_type = ObjectType.RIGID

Once registered, the object is available in the registry like any other asset:

.. code-block:: python

   obj = asset_registry.get_asset_by_name("my_object")()

Assets can also be tagged to make them discoverable by category:

.. code-block:: python

   # All graspable objects
   objects = asset_registry.get_assets_by_tag("graspable")

   # A random graspable object
   obj = asset_registry.get_random_asset_by_tag("graspable")()

Useful tags include ``"graspable"``, ``"openable"``, ``"pressable"``, and ``"background"``.
Assets can have multiple tags — for example, a fruit is tagged both ``"graspable"`` and ``"food"``.

Object types
------------

Every asset has an object type that determines how it is simulated:

- **RIGID** — a single rigid body (boxes, bottles, tools, furniture).
- **ARTICULATION** — a multi-body object with joints (robots, doors, drawers, appliances).
- **BASE** — no physics; used for static backgrounds and markers.

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

Resetting nested background physics
-----------------------------------

Backgrounds are registered as ``BASE`` assets, but their composed USDs may contain
dynamic rigid bodies and articulations. Set ``reset_nested_physics=True`` on a
``Background`` to register those nested physics roots as private Isaac Lab scene
entities. Arena snapshots their initialized poses, velocities, and joint states and
restores them on each episode reset without making the bodies kinematic.

Explicit ``ObjectReference`` entries take ownership of their paths and are not
duplicated. Articulation links are reset through their articulation root, while
joint-connected rigid bodies without an articulation root are reset individually.

Discovery follows the composed USD physics APIs rather than asset naming conventions.
For Replicator content, physics contributed by a resolved ``_OnlineVisual`` reference
is included. A ``_PlacementEnvelope`` is excluded when it is guide geometry without
physics APIs. Instanceable subtrees that contribute dynamic physics are materialized
at spawn time because physics views cannot control dynamic instance proxies.

Rigid object sets
-----------------

To fill one scene role with different rigid objects across parallel
environments, wrap the candidates in a ``RigidObjectSet``. See
:doc:`./concept_rigid_object_set` for motivation, usage, and limitations.
