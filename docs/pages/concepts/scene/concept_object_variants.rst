Per-Environment Objects
=======================

An ``Object`` uses the same asset in every parallel environment. A
``PerEnvironmentObject`` assigns one of several rigid objects to each environment.
Relations, tasks, and contact sensors address the named scene entry in either case.

Declare the alternatives
-------------------------

.. tab-set::

   .. tab-item:: Python
      :selected:

      Use a library object directly for the same asset in every environment.
      To distribute different assets, pass library objects to ``PerEnvironmentObject``:

      .. code-block:: python

         from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject

         banana = asset_registry.get_asset_by_name("banana_ycb_robolab")()
         orange = asset_registry.get_asset_by_name("orange_01_fruits_veggies_robolab")()
         fruit = PerEnvironmentObject(
             name="fruit",
             objects=[banana, orange],
             assign_to_environments="sequential",
         )

      Members' native spawn settings, including scale and physics, are copied.
      Configure the scene name, initial pose, and relations on ``fruit``; these
      are not copied from members. Native Isaac Lab spawn configurations are
      also accepted in ``objects``.

   .. tab-item:: YAML

      Declare ordinary assets under ``objects`` and ``PerEnvironmentObject``
      entries under ``per_environment_objects``:

      .. code-block:: yaml

         objects:
         - id: bowl
           registry_name: bowl_ycb_robolab
         per_environment_objects:
         - id: fruit
           objects:
           - registry_name: banana_ycb_robolab
           - registry_name: orange_01_fruits_veggies_robolab
             params:
               scale: [0.8, 0.8, 0.8]
           assign_to_environments: sequential

      Relations and tasks reference ``fruit``. Entry-level ``params`` configure
      its initial pose; member ``params`` configure individual assets, including
      scale or a SimReady ``usd_path``. The former ``object_sets`` section is rejected.

Assign assets to environments
-----------------------------

``ArenaEnvBuilder`` assigns assets once during environment construction, before
placement. Placement and Isaac Lab spawning use the same assignment, which stays
fixed across resets even when the layout changes.

- ``assign_to_environments="sequential"`` (default) cycles through the declared
  order: banana, orange, banana, orange.
- ``assign_to_environments="random"`` samples each environment independently,
  so repeats are possible. The seed comes from the builder's ``placement_seed``,
  then ``arena_env.placer_params.placement_seed`` if unset, then the builder's ``seed``.

These are Arena settings applied to each ``PerEnvironmentObject`` independently.
Isaac Lab's scene-level ``clone_strategy`` instead accepts callables that assign
combinations across the scene. When calling a placement solver directly, call
``assign_object_variants(scene.assets.values(), num_envs=4, seed=42)`` from
``isaaclab_arena.scene.object_variant_assignment`` first.

Native spawning and asset preparation
---------------------------------------

Multiple assets use Isaac Lab's ``MultiAssetSpawnerCfg``, accessible through
``fruit.spawn_cfg.assets_cfg``; a single asset uses its concrete ``fruit.spawn_cfg``.
``fruit.asset_indices_by_env`` identifies the selected configuration per environment.
Each asset's scale stays native. Rigid-body views and contact sensors require a
common relative body path, so incompatible USD hierarchies receive cached prepared
copies. Compatible assets retain their original sources; source files remain unchanged.

Constraints
------------

- Alternatives must be rigid objects with exactly one rigid body each.
  Articulations, empty object lists, and nested ``PerEnvironmentObject`` entries
  are rejected.
- With multiple alternatives, construct a new ``PerEnvironmentObject`` to change
  the assignment or environment count.
- Use ``get_bounding_box_per_env()`` with multiple alternatives;
  ``get_bounding_box()`` requires a single asset.
- Heterogeneous placement uses per-environment axis-aligned bounding boxes even
  in mesh collision mode. Simulation uses each asset's configured collision geometry.
- A fixed heterogeneous obstacle needs an ``IsAnchor()`` relation and a known
  initial pose; relation-free passive obstacles require shared geometry.
- Native primitive spawners can participate when their relative body paths are
  compatible. Preparing different body hierarchies requires USD sources.
- Do not set an initial pose when placement relations determine the object's
  pose; anchors require a known pose.

See :doc:`../object_placement/homogeneous_and_heterogeneous_placement` for
placement examples and :doc:`../object_placement/pooled_placement` for resets.
