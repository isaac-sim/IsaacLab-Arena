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

First-party environment-specific asset classes can live in
``isaaclab_arena_environments/<environment>/assets.py``. Normal environment
registration imports the package-level ``asset_library.py`` entry point, which
imports these modules and runs their ``@register_asset`` decorators. Only these
asset modules are discovered; nested prebuild and runtime helper scripts are not
imported. Keep acquisition and derivation lazy so registration does not download
or build USD files.

Python callers should call
``isaaclab_arena_environments.cli.ensure_environments_registered()`` before looking
up environment-specific assets. The graph-spec and policy runners already use this
registration path. For example, the workbench classes live alongside their recipes
in ``isaaclab_arena_environments/workbench_cleanup/assets.py``; their existing
``workbench_cleanup_*`` registry names remain valid in YAML.

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

Cached USD derivation
~~~~~~~~~~~~~~~~~~~~~

Use ``DerivedUsdSource`` for edits that must be baked into a reusable USD package,
such as mesh partitions, fixed articulations, joint drives, or material overrides.
Pass a local entry path directly, or use an input ``UsdSource`` to acquire the package. Object type,
tags, affordances, and initial scene state remain in the registered asset class.

.. code-block:: python

   from isaaclab_arena.assets.sources.derived import DerivedUsdSource

   source = DerivedUsdSource(
       "/data/package/usd/item.usda",
       "/data/derivation.json",
   )
   derived_usd_path = source.resolve()

For a direct path, the default source identity is the path string. Sources that expose
``identity()`` use that value. For a source implementing only ``resolve()``, pass
``source_identity="<stable-package-identity>"`` to ``DerivedUsdSource``. The identity
must match the recipe's ``source_identity`` and is available without acquiring the package.

The implementation lives under ``isaaclab_arena/assets/derivation/``.
The source adapter delegates to its cache engine, which copies the complete source
package, applies derivation in a staging directory, and atomically publishes the
result under ``~/.cache/isaaclab_arena/assets/derived``. A cache hit verifies the
manifest's file hashes and does not resolve or download the source again. The source identity, recipe bytes, and pipeline version
form the cache key. Bump ``DERIVATION_PIPELINE_VERSION`` when changing derivation
behavior without changing its recipe.

Recipes are JSON files read from a filesystem path or a package resource such as
``package://isaaclab_arena_environments/workbench_cleanup/assets/recipes/white_bin.json``.
Each recipe declares ``source_identity``, ``source_entry``, ``output_entry``, and
``expected_source_sha256``. Entry paths are relative to the source package root;
source and output must be different files. The workbench recipes demonstrate
pinned complete-package acquisition with ``SimReadyCentralPackageSource``.

Set ``operation`` to
``isaaclab_arena.assets.derivation.pipeline:derive_asset`` and provide an
ordered ``steps`` list. These steps depend only on USD, with no environment imports:

.. list-table::
   :header-rows: 1

   * - Operation
     - Configuration
   * - ``partition_mesh``
     - ``prim``, named component ``groups``, optional ``remainder`` and ``expected_geometry_sha256``.
   * - ``fixed_articulations``
     - Named ``groups`` containing ``frame_mesh`` and existing ``joints`` paths.
   * - ``edit_prims``
     - ``paths``, optional ``descendants``, ``has_api``, ``is_a``, ``is_instanceable``,
       and ``exclude_paths`` selectors; API additions/removals, typed attributes,
       deinstancing, type-name edits, or activation changes.
   * - ``joint_drive``
     - Joint ``path``, ``linear``/``angular`` ``instance``, and ``settings`` for drive
       type, target position, stiffness, damping, or maximum force, in USD units.
   * - ``fixed_joint``
     - New joint ``path``, existing ``body1``, and optional ``body0`` (omitted for world attachment).
   * - ``preview_surface``
     - Material ``path``, typed ``inputs``, optional ``bind_to`` prims, and
       ``stronger_than_descendants`` binding strength.

For example, an asset-independent material step is:

.. code-block:: json

   {
     "operation": "preview_surface",
     "path": "Materials/Plastic",
     "inputs": {
       "diffuseColor": {"type": "Color3f", "value": [0.85, 0.85, 0.85]},
       "roughness": {"type": "Float", "value": 0.5}
     },
     "bind_to": ["."],
     "stronger_than_descendants": true
   }

Paths refer to the default prim (``.``) or its descendants. Attribute and shader
input ``type`` values use ``Sdf.ValueTypeNames`` names; JSON arrays are converted
to the corresponding USD vector or array type. Attribute specifications may set
``custom: false`` for schema-defined properties. Only matching existing prims are
edited; missing explicitly selected paths are errors.

The wrapper preserves the source's units and up axis. Optional ``variants`` select
authored variants, and ``scale`` bakes a transform under ``BakedScale``; step paths
must account for that extra prim. ``root_prim`` defaults to ``/Asset``. Use
``required_prims`` to validate the result and ``flatten: true`` when a self-contained
layer is required. References and dependencies within the copied package remain
relative so the package can be relocated.

Keep asset-specific mesh paths, component IDs, hashes, physical tuning, and material
values in recipes. A new asset using these operations needs no derivation Python
in its environment package. Custom ``module:function`` operations remain supported
by the cache engine for edits that the shared pipeline cannot express.

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

Use ``ArticulationObjectReference`` to configure initial joints on a referenced articulation:

.. code-block:: python

   from isaaclab_arena.assets.object_reference import ArticulationObjectReference

   microwave = ArticulationObjectReference(
       name="microwave",
       prim_path="{ENV_REGEX_NS}/kitchen/microwave",
       parent_asset=kitchen,
       initial_joint_positions={"microjoint": -0.5},
   )

Joint keys may be exact names or Isaac Lab regex patterns. Values are radians for revolute joints
and metres for prismatic joints. ``OpenableObjectReference``, ``PressableObjectReference``, and
``TurnableObjectReference`` inherit this option. In graph YAML, the existing ``object_type: articulation`` and
``params.initial_joint_positions`` fields select the class automatically.

A reference can also name a link of an ``ARTICULATION`` object, such as a door or a drawer.
When a revolute or prismatic joint moves the link, the reference is placed where the
parent's configured initial joint positions put it, and during simulation it follows the link.
Create the articulation reference before its link references so they use that initial joint
state. Links inside a ``BASE`` background, such as a kitchen's cabinet doors, keep their authored
poses.

.. code-block:: python

   door = ObjectReference(
       name="microwave_door",
       prim_path="{ENV_REGEX_NS}/kitchen/microwave/Microwave039_door",
       parent_asset=microwave,
   )

Rigid object sets
-----------------

To fill one scene role with different rigid objects across parallel
environments, wrap the candidates in a ``RigidObjectSet``. See
:doc:`./concept_rigid_object_set` for motivation, usage, and limitations.
