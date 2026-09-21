Asset physics overrides
=======================

``PhysicsUsdFileCfg`` applies physics to exact colliders or joints inside one spawned USD
without modifying the source asset. The overrides are authored before cloning and physics
backend import, so other objects or embodiments that use the same USD keep their defaults.

Asset physics is Python configuration owned by the component that selects the asset. Do not
put ``PhysicsUsdFileCfg`` or ``PrimPhysicsCfg`` under graph YAML ``env_cfg_override``:

- Put robot physics shared by an embodiment in that embodiment's scene configuration. For a
  backend-specific override, apply it in ``EmbodimentBase._configure_physics_backend``.
- Put object physics on the object's ``spawner_cfg``. A registered object can construct that
  spawner in its class so every instance receives the same physics.

Embodiment-owned overrides
--------------------------

An embodiment owns its robot spawn configuration, actuator configuration, and control interface.
Apply robot contact or joint overrides there so the robot remains self-contained when combined
with different scenes and tasks. Use ``_configure_physics_backend`` when the configuration is
specific to PhysX or Newton:

.. code-block:: python

   from isaaclab.sim import UsdFileCfg
   from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonMaterialPropertiesCfg

   from isaaclab_arena.assets.physics_config import PhysicsUsdFileCfg, PrimPhysicsCfg
   from isaaclab_arena.utils.physics_backend import PhysicsBackend


   class InsertionEmbodiment(MyEmbodimentBase):
       def _configure_physics_backend(self, backend: PhysicsBackend) -> None:
           super()._configure_physics_backend(backend)
           if backend is not PhysicsBackend.NEWTON:
               return

           spawn = self.scene_config.robot.spawn
           assert isinstance(spawn, UsdFileCfg)
           self.scene_config.robot.spawn = PhysicsUsdFileCfg(
               usd_path=spawn.usd_path,
               scale=spawn.scale,
               make_uninstanceable=True,
               prim_physics={
                   "hand/finger/collision": PrimPhysicsCfg(
                       collision_props=[MujocoCollisionCfg(condim=4)],
                       physics_material=NewtonMaterialPropertiesCfg(
                           static_friction=8.0,
                           dynamic_friction=8.0,
                       ),
                   ),
               },
           )

Keep controlled-joint gains in the embodiment's actuator configuration. Articulation
initialization may overwrite drive values authored through ``joint_drive_props``.

Replacing an existing spawn configuration does not copy its options automatically. Preserve
every required field, such as scale, variants, visibility, and contact-sensor activation, when
constructing ``PhysicsUsdFileCfg``.

Object-owned overrides
----------------------

Pass ``PhysicsUsdFileCfg`` as an ``Object``'s ``spawner_cfg`` when the physics belongs to that
object rather than to the whole environment:

.. code-block:: python

   from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonMaterialPropertiesCfg

   from isaaclab_arena.assets.object import Object
   from isaaclab_arena.assets.object_type import ObjectType
   from isaaclab_arena.assets.physics_config import PhysicsUsdFileCfg, PrimPhysicsCfg


   connector = Object(
       name="connector",
       prim_path="{ENV_REGEX_NS}/Connector",
       object_type=ObjectType.RIGID,
       spawner_cfg=PhysicsUsdFileCfg(
           usd_path="/path/to/connector.usda",
           prim_physics={
               "Geometry": PrimPhysicsCfg(
                   collision_props=[
                       MujocoCollisionCfg(
                           solref=(0.004, 1.0),
                           solimp=(0.95, 0.999, 0.0005, 0.5, 2.0),
                       ),
                   ],
                   physics_material=NewtonMaterialPropertiesCfg(
                       static_friction=0.35,
                       dynamic_friction=0.35,
                       contact_stiffness=62500.0,
                       contact_damping=500.0,
                   ),
               ),
           },
       ),
   )

Use the same pattern inside a registered ``Object`` subclass when all instances of that asset
need the override. If an object already uses a custom spawn function, integrate
``apply_prim_physics`` into that spawner before cloning instead of replacing the function.

Supported per-prim properties
-----------------------------

``prim_physics`` maps exact paths relative to the spawned asset root to ``PrimPhysicsCfg``.
It supports:

- ``collision_props``: Isaac Lab collision fragments on a geometry prim. This can enable a
  collider on an authored visual mesh without adding procedural geometry.
- ``physics_material``: a material created and bound to the selected collider without editing
  a material shared with other asset parts or instances.
- ``joint_drive_props``: Isaac Lab joint-drive fragments on a revolute or prismatic joint.
- ``mujoco_equality``: ``solref`` and ``solimp`` on an existing ``MjcEqualityJointAPI``,
  ``MjcEqualityConnectAPI``, or ``MjcEqualityWeldAPI`` constraint. This is specific to Newton's
  MuJoCo solver and does not create or convert mimic constraints.
- ``filtered_pairs``: additional asset-relative rigid-body or collider exclusions. Existing
  exclusions are preserved and internal relationships are remapped during cloning.

All paths are validated before any override is authored. Missing or absolute paths, paths
outside the asset, incompatible prim types, and instance proxies are rejected. Use
``make_uninstanceable=True`` when selected prims are inside an instanceable USD subtree and the
additional stage memory is acceptable. Use ``.`` to select the asset root. Regular expressions
and wildcards are not supported.
