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

Physics spawn addons
--------------------

Use ``spawn_cfg_addon`` on scene objects to supply ordinary USD spawn options such as
``collision_props`` and ``physics_material``. To configure selected colliders or joints within an object, add
a ``prim_physics`` mapping. Arena applies these typed settings through ``with_spawn_cfg_addon()``,
the same helper used by embodiment addons. Ordinary fields replace the corresponding spawn
options; per-prim entries replace the settings for their named prims and retain other entries.
The object's USD path, scale, contact-sensor activation, and other spawn options are retained.
Define a concrete ``PrimPhysicsCfg`` subclass in the environment or use-case module that
needs it. Core defines only the interface; the subclass chooses its fields, validation,
and physics schema edits. For example, define a collider friction override in your
environment's physics configuration module and use it with the library's red cube:

.. code-block:: python

   import math

   from isaaclab.utils.configclass import configclass
   from pxr import UsdPhysics, UsdShade

   from isaaclab_arena.assets.object_library import RedCube
   from isaaclab_arena.assets.physics_config import PrimPhysicsCfg

   @configclass
   class ColliderFrictionCfg(PrimPhysicsCfg):
       friction: float = 0.8
       """Static and dynamic friction coefficient."""

       def validate_target(self, prim, root):
           assert prim.HasAPI(UsdPhysics.CollisionAPI)
           assert math.isfinite(self.friction) and self.friction >= 0
           assert not prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild("ContactMaterial"))

       def apply(self, prim, root):
           # Create an instance-local material so shared USD materials stay unchanged.
           material = UsdShade.Material.Define(
               prim.GetStage(), prim.GetPath().AppendChild("ContactMaterial")
           )
           physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
           physics.CreateStaticFrictionAttr(self.friction)
           physics.CreateDynamicFrictionAttr(self.friction)
           UsdShade.MaterialBindingAPI.Apply(prim).Bind(
               material,
               bindingStrength=UsdShade.Tokens.strongerThanDescendants,
               materialPurpose="physics",
           )

   class HighFrictionRedCube(RedCube):
       spawn_cfg_addon = {
           "prim_physics": {"Cube": ColliderFrictionCfg(friction=0.8)},
       }

   red_cube = HighFrictionRedCube()

The subclass inherits the library object's USD path and scale. ``Cube`` is the collider mesh
relative to the red cube's asset root. For other assets, use their exact relative prim paths, or
``"."`` for the root itself. Paths cannot be absolute, escape the asset, or contain wildcards.
Selected prims must already exist. Instance proxies require ``make_uninstanceable=True``
in the spawn addons, at the cost of additional stage memory.

Use concrete subclasses of ``PrimPhysicsCfg``; its base ``apply`` raises ``NotImplementedError``.
Implement ``apply(prim, root)`` and optionally ``validate_target(prim, root)``. Both receive the resolved target and spawned asset
root, allowing an implementation to resolve relationships within that asset. The optional
validation hook must be read-only and checks the stage before any per-prim overrides. It is
separate from Isaac Lab's configclass ``validate()`` method.

Arena resolves every target and calls every validation hook before calling ``apply`` in mapping
order. Validation failures therefore leave per-prim overrides unapplied; application itself is
not transactional. Concrete implementations must author within the spawned asset on the current
stage edit target, preserve source layers and shared materials, and validate any additional
relationship targets they use. Keep USD handles out of config fields so copying remains safe.

Ordinary USD spawn properties are applied first, then ``prim_physics``, then cloning and physics
model import. Concrete configs can implement collision, material, mass, joint, or backend-specific
settings as needed. Choose APIs compatible with the environment's physics backend. For controlled
joint gains, prefer actuator configuration because articulation initialization can overwrite USD drives.

A ``LibraryObject`` subclass can define the same dictionary as its ``spawn_cfg_addon`` class
attribute for shared defaults. Keep task-specific tuning in the environment's object/config
construction; composed spawn configs have independent copies of the physics settings.
An internal USD config subclass declares the extra field so Isaac Lab's ``copy()`` and
``replace()`` retain it. Object definitions only need the ``spawn_cfg_addon`` dictionary.

Robot and end-effector physics belong to the embodiment. Configure finger contact materials,
gripper colliders, collision exclusions, and coupling parameters in the embodiment's
``_configure_physics_backend()`` hook. See :doc:`../embodiment/index` for that configuration path.

With an explicit ``spawner_cfg``, put physics settings on that config instead of in
``spawn_cfg_addon``. Custom spawn functions must call
``isaaclab_arena.assets.physics_spawner.apply_prim_physics`` after creating the asset and before
cloning. Combining addon ``prim_physics`` with ``spawner_cfg`` or a custom addon ``func`` is
rejected so that a custom spawner cannot silently bypass the overrides.

Object types
------------

Every asset has an object type that determines how it is simulated:

- **RIGID** — a single rigid body (boxes, bottles, tools, furniture).
- **ARTICULATION** — a multi-body scene object with joints (doors, drawers, appliances).
- **BASE** — no physics; used for static backgrounds and markers.

Deformable and backend-specific spawn configs must match the environment's resolved physics
backend (PhysX or Newton). See :doc:`../environment/physics_backend_selection`.

Backgrounds
-----------

Backgrounds are registered as ``BASE`` assets, but their composed USDs may contain
dynamic rigid bodies and articulations whose states can change as they interact with
the robot or other objects. Arena resets these nested physics roots by default. Set
``reset_nested_physics=False`` on a ``Background`` to opt out.

Arena registers the roots as private Isaac Lab reset views. After simulation and RTX
initialization, Arena creates the views and records one environment-local pose and
joint configuration. On each episode reset, Arena applies those values to the
resetting environments and zeros all root and joint velocities. The private views
are not exposed through the Isaac Lab scene entity registries.

Arena discovers nested roots through the composed USD physics APIs.

Included:

- Standalone rigid bodies.
- Articulation roots, which own their links and joints.

Excluded:

- ``BASE`` and collision-only prims.
- Rigid and articulation roots owned by matching ``ObjectReference`` entries.
- Articulation links, which cannot own reset state independently.
- Authored roots without a live physics backend object after composition.

``BASE`` object references remain observational and do not transfer reset
ownership away from the background.

Instanceable subtrees that contribute dynamic physics are materialized at spawn
time because physics views cannot control dynamic instance proxies.

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

Rigid object sets
-----------------

To fill one scene role with different rigid objects across parallel
environments, wrap the candidates in a ``RigidObjectSet``. See
:doc:`./concept_rigid_object_set` for motivation, usage, and limitations.
