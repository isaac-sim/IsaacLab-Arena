Asset physics overrides
=======================

``PhysicsUsdFileCfg`` applies physics to selected prims in one spawned USD without changing
the source asset. Configure it in Python on the embodiment or object that owns the asset, not
under graph YAML ``env_cfg_override``.

API
---

``prim_physics`` maps exact asset-relative prim paths to ``PrimPhysicsCfg`` values:

.. code-block:: python

   from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonMaterialPropertiesCfg

   from isaaclab_arena.assets.physics_config import PhysicsUsdFileCfg, PrimPhysicsCfg


   def contact_spawn(usd_path: str, collider_path: str) -> PhysicsUsdFileCfg:
       return PhysicsUsdFileCfg(
           usd_path=usd_path,
           make_uninstanceable=True,
           prim_physics={
               collider_path: PrimPhysicsCfg(
                   collision_props=[
                       MujocoCollisionCfg(
                           solref=(0.004, 1.0),
                           solimp=(0.95, 0.999, 0.0005, 0.5, 2.0),
                       ),
                   ],
                   physics_material=NewtonMaterialPropertiesCfg(
                       static_friction=0.35,
                       dynamic_friction=0.35,
                   ),
               ),
           },
       )

``PrimPhysicsCfg`` supports ``collision_props``, ``physics_material``,
``joint_drive_props``, ``mujoco_equality``, and ``filtered_pairs``. Use ``.`` for the asset
root. Paths must be exact and cannot contain wildcards. Set ``make_uninstanceable=True`` when
the target is inside an instanceable subtree.

Embodiment override
-------------------

Attach robot physics to the embodiment's robot spawn configuration. Use
``_configure_physics_backend`` when the override is backend-specific:

.. code-block:: python

   def _configure_physics_backend(self, backend: PhysicsBackend) -> None:
       super()._configure_physics_backend(backend)
       if backend is PhysicsBackend.NEWTON:
           usd_path = self.scene_config.robot.spawn.usd_path
           self.scene_config.robot.spawn = contact_spawn(usd_path, "hand/finger/collision")

Keep controlled-joint gains in the embodiment's actuator configuration. When replacing an
existing spawn config, also preserve required options such as scale, variants, and visibility.

Object override
---------------

Attach object physics through ``spawner_cfg``:

.. code-block:: python

   connector = Object(
       name="connector",
       prim_path="{ENV_REGEX_NS}/Connector",
       object_type=ObjectType.RIGID,
       spawner_cfg=contact_spawn("/path/to/connector.usda", "Geometry"),
   )

A registered object can construct the same spawner in its class. If an asset already uses a
custom spawn function, call ``apply_prim_physics`` in that function before cloning.
