Object Choices
==============

An ``Object`` uses the same asset in every parallel environment. An
``ObjectChoice`` selects one of several rigid objects for each environment.
Relations, tasks, and contact sensors address the named scene entry in either case.

For example, a pick-and-place task can use an ``ObjectChoice`` named ``fruit``
that selects among a banana, orange, and lemon. Each environment manipulates
one fruit, while the task and placement relations are defined once.

Declare the alternatives
-------------------------

.. tab-set::

   .. tab-item:: Python
      :selected:

      Use a library asset directly when every environment needs the same asset:

      .. code-block:: python

         fruit = asset_registry.get_asset_by_name("banana_ycb_robolab")(
             instance_name="fruit"
         )

      For alternatives, pass library objects to ``ObjectChoice(objects=...)``.
      It copies their native spawn settings, including scale and physics. Their
      scene names, initial poses, and relations are not copied; these belong to
      ``fruit``.

      .. code-block:: python

         from isaaclab_arena.assets.object_choice import ObjectChoice
         from isaaclab_arena.relations.relations import On
         from isaaclab_arena.scene.scene import Scene
         from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

         banana = asset_registry.get_asset_by_name("banana_ycb_robolab")()
         orange = asset_registry.get_asset_by_name("orange_01_fruits_veggies_robolab")()
         fruit = ObjectChoice(
             name="fruit",
             objects=[banana, orange],
             assign_to_environments="sequential",
         )
         fruit.add_relation(On(table_reference))

         scene = Scene(assets=[background, table_reference, bowl, fruit])
         task = PickAndPlaceTask(
             pick_up_object=fruit,
             destination_location=bowl,
             background_scene=background,
         )

      A choice containing one object gives each environment its own instance of
      that asset. Use the library object directly when no choice is needed.
      Advanced callers can also pass native Isaac Lab spawner configurations
      directly in ``objects``.

   .. tab-item:: YAML

      Both classes are declared under ``objects``. ``registry_name`` creates an
      ordinary ``Object``; ``variants`` creates an ``ObjectChoice``. Each variant
      accepts its own constructor parameters, including a SimReady ``usd_path``
      or scale:

      .. code-block:: yaml

         objects:
         - id: bowl
           registry_name: bowl_ycb_robolab
         - id: fruit
           variants:
           - registry_name: banana_ycb_robolab
           - registry_name: orange_01_fruits_veggies_robolab
             params:
               scale: [0.8, 0.8, 0.8]
           - registry_name: simready_usd_object
             params:
               usd_path: /datasets/lemon.usd
           assign_to_environments: sequential
         relations:
         - kind: 'on'
           subject: fruit
           reference: maple_table
         task:
           composition: atomic
           description: Place the fruit in the bowl.
           subtasks:
           - kind: PickAndPlaceTask
             params:
               pick_up_object: fruit
               destination_location: bowl
               background_scene: maple_table

      Parameters on the object entry configure the role, such as its initial
      pose. Parameters inside ``variants`` configure the individual assets.
      The former ``object_sets`` section is rejected.

Assign variants to environments
--------------------------------

``ArenaEnvBuilder`` assigns variants once during environment construction, before
placement, and passes that same assignment to Isaac Lab's native scene clone planning.
``assign_to_environments="sequential"`` is the default: it cycles through
the declared order, such as banana, orange, banana, orange.
``assign_to_environments="random"`` samples each environment independently,
so repeats are possible. To reproduce random choices, assignment uses the
builder's ``placement_seed``, then ``arena_env.placer_params.placement_seed``
when the builder value is unset, and finally the builder's ``seed``.
The assignment remains fixed across resets, even when the layout changes.

This setting applies to each ``ObjectChoice`` independently. Isaac Lab's scene-level
``clone_strategy`` instead assigns combinations of variants across the scene.

Assignments are stored as data in the environment configuration, so Isaac Lab's
Hydra configuration roundtrip preserves them. Arena installs the native clone
strategy when the environment starts and checks that configuration overrides
have not changed the assigned assets or environment count.

With two variants and ``--num_envs 2``, sequential assignment gives each environment
one different fruit. Use more than one environment to see alternatives side by
side.

If you call a placement solver directly instead of using ``ArenaEnvBuilder``,
establish the assignment first:

.. code-block:: python

   from isaaclab_arena.scene.object_variant_assignment import assign_object_variants

   assign_object_variants(scene.assets.values(), num_envs=4, seed=42)
   # The placement solver can now use each environment's selected geometry.

Native spawning and asset preparation
---------------------------------------

``ObjectChoice`` copies the native spawn configurations passed in ``objects``.
Multiple alternatives are stored in Isaac Lab's ``MultiAssetSpawnerCfg`` and are
available through ``fruit.spawn_cfg.assets_cfg``.
A single alternative uses its concrete native configuration directly as
``fruit.spawn_cfg``.
After assignment, ``fruit.variant_indices_by_env`` identifies the selected
configuration for each environment.

Each variant's scale stays in its native spawn configuration; scale differences
do not require rewriting USD files.

Isaac Lab's batched rigid-body views and contact sensors require a common
relative body path. ``prepare_rigid_object_variants()`` handles incompatible USD
hierarchies in a separate asset-preparation module. Compatible assets retain
their original sources. Incompatible assets receive cached prepared copies;
source files remain unchanged. The same prepared configurations serve spawning
and contact-body discovery.

Constraints
------------

- Alternatives must be rigid objects with exactly one rigid body each.
  Articulations, empty choices, and nested ``ObjectChoice`` entries are rejected.
- With multiple alternatives, selection is fixed for the scene's lifetime.
  Construct a new ``ObjectChoice`` for a different assignment or environment count.
- Use ``get_bounding_box_per_env()`` for an ``ObjectChoice`` with multiple
  alternatives. ``get_bounding_box()`` requires a single concrete variant; it
  does not choose an arbitrary representative.
- Placement uses per-environment axis-aligned bounding boxes for heterogeneous
  objects, including when mesh collision mode is selected. Per-variant mesh
  placement is not supported; simulation still uses each variant's configured
  collision geometry.
- A fixed heterogeneous obstacle needs an ``IsAnchor()`` relation and a known
  initial pose. Relation-free passive obstacles share one geometry across
  environments, so they cannot represent different variants. Marking the object
  as an anchor makes placement use its assigned geometry in each environment.
- Native primitive spawners can participate when their relative body paths are
  compatible. Preparing different body hierarchies requires USD sources.
- Do not set an initial pose when placement relations determine the object's
  pose. Anchors remain fixed and require a known pose.

See :doc:`../object_placement/homogeneous_and_heterogeneous_placement` for
placement examples and :doc:`../object_placement/pooled_placement` for resets.
