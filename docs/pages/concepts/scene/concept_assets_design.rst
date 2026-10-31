Assets
======

An asset is anything Arena places in a scene, such as a cracker box, a kitchen background,
a robot, or a light. Each asset has a ``name``, which identifies it in the environment:
variation keys such as ``<name>.mass``, its prim path, its contact sensor, and recorded data
use it. Most assets are objects (``Object``): spawned from a USD file or a spawn configuration
as a rigid, articulated, or static body. A robot (an embodiment) is an asset but not an object.

Arena ships with a library of asset classes, and new ones can be added by registering them.
Each asset class is registered under a ``registry_name``, such as ``rubiks_cube_hot3d_robolab``.
``asset_registry.get_asset_by_name()`` returns the class. Calling the class creates an asset.
A library object such as ``CrackerBox`` is named after its registry name unless you pass ``name``,
for example to name it after its role in the task:

.. code-block:: python

   CrackerBox = asset_registry.get_asset_by_name("cracker_box")  # the class, registry_name "cracker_box"
   cracker_box = CrackerBox()  # an asset named "cracker_box"
   pick_up_object = CrackerBox(name="pick_up_object")  # an asset named "pick_up_object"
   pick_up_object.registry_name  # "cracker_box"

Backgrounds and robots take no ``name`` argument. Their ``name`` is always their registry name.

Most code does both steps in one line:

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   cracker_box = asset_registry.get_asset_by_name("cracker_box")()

Registering a new asset
-----------------------

To add a new object, subclass ``LibraryObject``, set its ``registry_name``, provide the USD path
and object type, and decorate it with ``@register_asset``:

.. code-block:: python

   @register_asset
   class MyObject(LibraryObject):
       registry_name = "my_object"
       tags = ["object", "graspable"]
       usd_path = "path/to/my_object.usd"
       object_type = ObjectType.RIGID

Once registered, the class is available in the registry like any other asset class:

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

Object configuration
--------------------

An object's constructor sets its USD source, scale, initial pose, and object type.
``asset_cfg_addon`` configures the Isaac Lab asset (for example, ``debug_vis`` or articulation
actuators). ``spawn_cfg_addon`` configures how the USD is loaded and which physics properties
are authored during spawning: mass/density, collision settings, and contact materials.
Use ``prim_physics`` within the spawn addons for selected bodies, colliders, or joints.

Physics spawn addons
~~~~~~~~~~~~~~~~~~~~

Use ``spawn_cfg_addon`` to override physics parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Addon
     - Example parameters
   * - ``mass_props``
     - Mass and density.
   * - ``rigid_props``
     - Gravity, damping, and velocity limits.
   * - ``collision_props``
     - Collision enablement, contact offset, and rest offset.
   * - ``physics_material``
     - Static/dynamic friction and restitution.
   * - ``articulation_props``
     - Self-collision and articulation solver settings.
   * - ``prim_physics``
     - Selected-prim collision, material, mass, joint, or backend-specific settings,
       defined by an environment-owned ``UsdPrimSpawnPhysicsCfg`` subclass.

For example, override friction on the library red cube's ``Cube`` collider:

.. code-block:: python

   import math

   from isaaclab.utils.configclass import configclass
   from pxr import UsdPhysics, UsdShade

   from isaaclab_arena.assets.object_library import RedCube
   from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg

   @configclass
   class ColliderFrictionCfg(UsdPrimSpawnPhysicsCfg):
       """Bind an instance-local contact material to a selected collider."""

       friction: float = 0.8
       """Static and dynamic friction coefficient."""

       def validate_target(self, prim, root):
           """Require a collider and a finite nonnegative friction coefficient."""
           assert prim.HasAPI(UsdPhysics.CollisionAPI)
           assert math.isfinite(self.friction) and self.friction >= 0
           assert not prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild("ContactMaterial"))

       def apply(self, prim, root):
           """Author and bind a material within the spawned asset."""
           # A local material preserves shared source materials and remaps during cloning.
           material = UsdShade.Material.Define(prim.GetStage(), prim.GetPath().AppendChild("ContactMaterial"))
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

``prim_physics`` keys are exact paths relative to the asset root; use ``"."`` for the root.
Settings apply after USD loading and before cloning and physics import.

See :doc:`../environment/physics_configuration` for configuration order and
:doc:`../embodiment/index` for robot and end-effector physics.

SimReady USD assets
-------------------

Arena provides generic registered classes for loading SimReady USDs directly from an
Environment Graph Spec. Use these classes when the asset is not already represented by a
more specific Arena library class:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Registry name and class
     - Purpose
   * - ``simready_usd_object`` (``SimReadyUsdObject``)
     - One rigid object.
   * - ``simready_usd_articulation`` (``SimReadyUsdArticulation``)
     - A generic articulated object.
   * - ``simready_usd_openable_articulation`` (``SimReadyUsdOpenableArticulation``)
     - An articulation whose configured joint provides the ``Openable`` affordance.
   * - ``simready_usd_pressable_articulation`` (``SimReadyUsdPressableArticulation``)
     - An articulation whose configured joint provides the ``Pressable`` affordance.
   * - ``simready_usd_background`` (``SimReadyUsdBackground``)
     - A scene background, optionally converted to static collision and augmented with
       reference colliders.

The ``usd_path`` may be a supported local path or remote URL. The following rigid objects
use the beverage can and basket from the agentic environment-generation workflow:

.. code-block:: yaml

   objects:
   - id: beverage_can
     registry_name: simready_usd_object
     params:
       usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady/Residential/Kitchen/Food/Canned_Goods/Can_M01/sm_food_beverage_can_m01_01.usd
       scale: [1.0, 1.0, 1.0]
   - id: mini_plastic_basket
     registry_name: simready_usd_object
     params:
       usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady/Residential/Kitchen/Baskets/Plastic_Basket_A01/sm_misc_basket_plastic_a01_01.usd

``SimReadyUsdObject`` selects the asset's ``Physics=physics`` variant by default. Inspect the
USD before configuring it: the generic class expects exactly one rigid body, while assets with
joints must use one of the articulation classes.

Overriding rigid-body and collision properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass common Isaac Lab spawn settings through ``spawn_cfg_addon``. For graph YAML,
``SimReadyUsdObject`` converts mappings for ``rigid_props`` and ``collision_props`` to
``RigidBodyPropertiesCfg`` and ``CollisionPropertiesCfg``:

.. code-block:: yaml

   objects:
   - id: beverage_can
     registry_name: simready_usd_object
     params:
       usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady/Residential/Kitchen/Food/Canned_Goods/Can_M01/sm_food_beverage_can_m01_01.usd
       spawn_cfg_addon:
         make_uninstanceable: true
         rigid_props:
           linear_damping: 2.0
           angular_damping: 2.0
         collision_props:
           contact_offset: 0.001
           rest_offset: 0.0
           mesh_collision_property:
             mesh_approximation_name: convexHull

Only specify fields accepted by the corresponding Isaac Lab configuration class. These
overrides apply to the spawned asset without modifying the source USD.

Re-rooting and fixing an articulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SimReadyUsdArticulation`` can generate a cached wrapper that moves
``UsdPhysics.ArticulationRootAPI`` and attaches one authored rigid body to the world:

.. code-block:: yaml

   objects:
   - id: cabinet
     registry_name: simready_usd_articulation
     params:
       usd_path: <articulated-simready-usd-path>
       articulation_root_prim_path: .
       fixed_base_body_path: Geometry/base
       initial_joint_positions:
         drawer_joint: 0.2

``articulation_root_prim_path`` and ``fixed_base_body_path`` are paths relative to the source
asset's default prim. Setting ``articulation_root_prim_path`` removes existing articulation-root
APIs below the selected root and applies the API to that root. Setting
``fixed_base_body_path`` creates a world fixed joint targeting that rigid body. The wrapper
references the original USD; it does not edit it. Verify both paths and the joint name in the
composed USD. Re-rooting is needed when the source authors
``UsdPhysics.ArticulationRootAPI`` on a nested prim that does not match the prim spawned and
matched by Isaac Lab's articulation view. Isaac Lab does not generally require the articulation
root prim itself to be a rigid body. The fixed-base path is separate: it must identify an
authored rigid body because the generated world fixed joint attaches to that body, preventing
the articulation base from moving.

Use the affordance-specific class when a task needs to identify an interaction joint:

.. code-block:: yaml

   objects:
   - id: cabinet
     registry_name: simready_usd_openable_articulation
     params:
       usd_path: <articulated-simready-usd-path>
       articulation_root_prim_path: .
       fixed_base_body_path: Geometry/base
       openable_joint_name: drawer_joint
       initial_joint_positions:
         drawer_joint: 0.2

For a button, use ``simready_usd_pressable_articulation`` and
``pressable_joint_name`` instead. Re-rooting is optional; use it only when the source asset's
articulation root is absent or unsuitable for Isaac Lab's articulation view.

Adding a reference collider to a background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SimReadyUsdBackground`` accepts ``reference_colliders``. Each entry creates an invisible,
axis-aligned box collider in the generated wrapper. This is useful when an environment needs a
simple, stable placement or task surface that the source USD does not expose as a separate prim.
Most commonly, the desired surface is only one subregion of a larger authored collider mesh.
Position and size the reference collider so its top surface aligns with the corresponding surface
of the original mesh rather than creating new or visibly offset collision geometry:

.. code-block:: yaml

   background:
     id: work_surface
     registry_name: simready_usd_background
     params:
       usd_path: <simready-background-usd-path>
       object_min_z: 0.0
       static_collision_only: true
       reference_colliders:
         PlacementSurface:
           position_xyz: [0.0, 0.0, 0.75]
           size_xyz: [1.0, 0.6, 0.01]
   object_references:
   - id: placement_surface
     parent_id: work_surface
     prim_path: PlacementSurface
     object_type: base
     params: {}

The map key is an asset-relative prim path. Its ``position_xyz`` is the box center and
``size_xyz`` contains three positive full extents, all in the asset frame. The
``ObjectReference`` named ``placement_surface`` makes that generated collider available to
relations and tasks. ``static_collision_only: true`` also removes rigid-body and articulation
APIs and disables joints throughout the background, so enable it only when no nested body needs
to move. ``object_min_z`` is the background's local minimum height used for initial placement.

Object types
------------

Every asset has an object type that determines how it is simulated:

- **RIGID** — a single rigid body (boxes, bottles, tools, furniture).
- **ARTICULATION** — a multi-body scene object with joints (doors, drawers, appliances).
- **DEFORMABLE** — a body represented by deformable simulation nodes (soft cubes, cloth,
  or other compliant objects).
- **BASE** — no physics; used for static backgrounds and markers.

Deformable and backend-specific spawn configs must match the environment's resolved physics
backend (PhysX or Newton). See :doc:`../environment/physics_backend_selection`.

Deformable objects
------------------

``DeformableObject`` represents a body whose geometry changes during simulation. Its spawn
configuration selects the physics implementation: ``PhysxDeformableBodyPropertiesCfg`` for
PhysX or ``NewtonDeformableBodyPropertiesCfg`` for Newton. The selected implementation must
match the environment's physics backend.

Newton deformables additionally require a Vertex Block Descent (VBD) solver that owns their
particles. ``DeformableObject.validate_simulation_cfg()`` accepts either a direct
``VBDSolverCfg`` or a coupled solver with a particle-owning VBD entry. It rejects Newton
configurations that provide only ``MJWarpSolverCfg`` or that include a VBD entry without
assigning particles. This validation runs after the environment's ``env_cfg_callback``, so
an environment can install its coupled rigid-body and deformable solvers before Arena checks
the final simulation configuration.

Arena uses two complementary geometry representations for a deformable:

* Before simulation, ``get_bounding_box()`` returns the undeformed, local-frame bounds from
  the spawn configuration or USD. Relation placement uses these rest bounds to sample an
  initial pose without requiring a running physics view.
* During simulation, ``ArenaWorld`` reads the live deformable state. It exposes the current
  nodal vertices, centroid, world bounds, and mean nodal velocity through the same geometry
  query interface used by task predicates.

Predicates therefore evaluate the deformed object rather than its rest bounds. For example,
pick-and-place checks use the live centroid and velocity, while deformable support checks use
the lowest live nodes near the destination geometry. Deformables do not currently provide an
Isaac Lab contact sensor, so predicates that support them use these geometry queries.

The following image shows four initial placements sampled from the same ``On`` and ``NextTo``
relations. The undeformed cube bounds keep every sample on the table and beside the plate;
the live nodal geometry takes over after simulation starts.

.. image:: ../../../images/deformable_pick_and_place_sampled_placements.png
   :align: center
   :alt: Four relation-sampled placements of a deformable cube beside a plate

For a complete task and backend-specific commands, see
:doc:`../../example_workflows/newton/deformable_pick_and_place`.

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
