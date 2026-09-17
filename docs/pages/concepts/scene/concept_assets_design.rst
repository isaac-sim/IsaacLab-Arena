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

Use ``spawn_cfg_addon`` to supply ordinary USD spawn options such as ``collision_props``
and ``physics_material``. To configure selected colliders or joints within an asset, add
a ``prim_physics`` mapping. Arena then selects ``PhysicsUsdFileCfg`` automatically, retaining
the object's USD path, scale, contact-sensor activation, and other spawn options.

.. code-block:: python

   from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonMaterialPropertiesCfg

   from isaaclab_arena.assets.object import Object
   from isaaclab_arena.assets.object_type import ObjectType
   from isaaclab_arena.assets.physics_config import PrimPhysicsCfg

   robot = Object(
       name="robot",
       usd_path="/path/to/robot.usda",
       object_type=ObjectType.ARTICULATION,
       spawn_cfg_addon={
           "copy_from_source": False,
           "prim_physics": {
               "finger/collision": PrimPhysicsCfg(
                   collision_props=[
                       MujocoCollisionCfg(condim=4, solref=(0.004, 1.0)),
                   ],
                   physics_material=NewtonMaterialPropertiesCfg(
                       static_friction=8.0, dynamic_friction=8.0,
                   ),
               ),
           },
       },
   )

The example path is illustrative: use exact prim paths relative to the asset root, or
``"."`` for the root itself. Paths cannot be absolute, escape the asset, or contain wildcards.
Selected prims must already exist. Instance proxies require ``make_uninstanceable=True``
in the spawn addons, at the cost of additional stage memory.

``PrimPhysicsCfg`` supports:

- ``collision_props``: Isaac Lab collision fragments, including collider enablement and
  backend-specific contact parameters.
- ``physics_material``: a material created and bound locally to the selected collider.
- ``joint_drive_props``: drive fragments on revolute or prismatic joints. Use actuator configs
  for controlled joint gains, since articulation initialization may overwrite authored drives.
- ``mujoco_equality``: ``MujocoEqualityPropertiesCfg(solref=..., solimp=...)`` to tune an existing
  MuJoCo equality constraint. It does not create a coupling or change its leader or coefficients.
- ``filtered_pairs``: additional asset-relative rigid-body or collider paths to exclude from
  collision. Existing exclusions are preserved.

Ordinary USD spawn properties are applied first, then ``prim_physics``, then cloning and physics
model import. Overrides affect the spawned instance without editing the source USD or shared
materials. Choose fragments compatible with the environment's physics backend.

A ``LibraryObject`` subclass can define the same dictionary as its ``spawn_cfg_addon`` class
attribute for shared defaults. Keep task-specific tuning in the environment's object/config
construction; composed spawn configs have independent copies of the physics settings.
Embodiments constructing ``ArticulationCfg`` directly can set ``spawn=PhysicsUsdFileCfg(...)``.

With an explicit ``spawner_cfg``, put physics settings on that config instead of in
``spawn_cfg_addon``. Custom spawn functions must call
``isaaclab_arena.assets.physics_spawner.apply_prim_physics`` after creating the asset and before
cloning. Combining addon ``prim_physics`` with either ``spawner_cfg`` or an addon ``func`` is
rejected so that a custom spawner cannot silently bypass the overrides.

Object types
------------

Every asset has an object type that determines how it is simulated:

- **RIGID** — a single rigid body (boxes, bottles, tools, furniture).
- **ARTICULATION** — a multi-body object with joints (robots, doors, drawers, appliances).
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
