Rigid Object Sets
=================

Use a ``RigidObjectSet`` to spawn **heterogeneous objects across parallel
environments** — different rigid objects per env, without changing the scene
or task definition.

Motivation
----------

A plain ``Object`` clones the same USD into every environment. To spawn a
different object per env without rewriting the environment definition, wrap the
candidates in a ``RigidObjectSet``. Arena assigns one member per environment
before spawn and uses that member's USD and bounding box for placement.

**Example.** Suppose the task is to pick a fruit and place it in a bowl. With a
single ``Object``, every parallel environment tests the same fruit. With a
``RigidObjectSet`` of banana, orange, and lemon, each environment can test a
different fruit — same pick-and-place task, richer coverage in one run.

What It Is
----------

``RigidObjectSet`` is a subclass of ``Object`` that holds a list of rigid
members. To the scene and task it still looks like one asset — one name, one
prim path, and (optionally) one set of relations. Only the spawned object
differs across environments.

When the environment is built and ``num_envs`` is known, Arena assigns one
member to each environment. That choice is locked for the run, so the spawned
USD and the geometry used by the placement solver stay aligned.

How members are chosen:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Behavior
   * - Ordered (default)
     - Environments cycle through the list: ``env_idx % len(objects)``.
   * - Random (``random_choice=True``)
     - Each environment picks independently. Pass a placement seed to make the
       choice reproducible.

Members must be rigid objects. Articulations and empty lists are rejected when
you construct the set.

How to Use
----------

Build the fruit members, wrap them in a set, add the set to the scene, and pass
it to the task as the pick object:

.. tab-set::

   .. tab-item:: Python
      :selected:

      .. code-block:: python

         from isaaclab_arena.assets.object_set import RigidObjectSet
         from isaaclab_arena.relations.relations import On
         from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

         members = [
             asset_registry.get_asset_by_name("banana_ycb_robolab")(),
             asset_registry.get_asset_by_name("orange_01_fruits_veggies_robolab")(),
             asset_registry.get_asset_by_name("lemon_01_fruits_veggies_robolab")(),
         ]
         fruit_set = RigidObjectSet(name="fruit", objects=members)
         fruit_set.add_relation(On(table_reference))

         scene = Scene(assets=[background, table, bowl, fruit_set])
         task = PickAndPlaceTask(
             pick_up_object=fruit_set,
             destination_location=bowl,
             background_scene=background,
         )

   .. tab-item:: YAML

      In an environment-graph spec, declare the set under ``object_sets`` and
      reference its ``id`` from relations and the task — the same way you would
      reference a single object:

      .. code-block:: yaml

         objects:
         - id: bowl
           registry_name: bowl_ycb_robolab
           params: {}
         object_sets:
         - id: fruit
           members:
           - banana_ycb_robolab
           - orange_01_fruits_veggies_robolab
           - lemon_01_fruits_veggies_robolab
           random_choice: false
           params: {}
         relations:
         - kind: 'on'
           subject: fruit
           reference: maple_table
           params: {}
         task:
           composition: atomic
           subtasks:
           - kind: PickAndPlaceTask
             params:
               pick_up_object: fruit
               destination_location: bowl
               background_scene: maple_table

With ``--num_envs 3`` and ordered assignment, one environment gets the banana,
one the orange, and one the lemon. The task definition stays the same; only the
spawned fruit differs.

A few things to know:

- **Set ``num_envs`` first.** Arena needs the environment count before it can
  assign members — in runners this is ``--num_envs``. Use more than one to see
  different members side by side.
- **Tasks take the set directly.** Anything that accepts an ``Object`` works;
  Arena resolves the active member through the shared prim path.

For a placement-focused comparison of same-object vs set-based roles, see
:doc:`../object_placement/homogeneous_and_heterogeneous_placement`.

Limitations
-----------

- **Rigid only.** Members must be ``ObjectType.RIGID`` — no articulations.
- **Choice is fixed per run.** Layouts can change on reset; the assigned member
  cannot.
- **One shared prim path.** Every environment uses ``{ENV_REGEX_NS}/<name>``.
  Arena may rewrite and cache member USDs so they spawn correctly together (for
  example when scales or rigid-body nesting differ).
- **Single-box geometry has a fallback.** ``get_bounding_box()`` returns one box
  from the tallest member; per-environment placement uses
  ``get_bounding_box_per_env()`` instead.
- **YAML members can't carry a ``usd_path``.** Graph YAML builds members with no
  constructor args, so SimReady assets that need a custom path can't be YAML set
  members. In Python you pass live instances, so SimReady members are fine.
- **Don't set an initial pose when placement owns it.** The builder supplies the
  set's creation and reset poses. Anchors are the exception — they stay fixed and
  still need a known pose.
- **Contact sensors use the first member.** Setup uses the canonical first-member
  USD after any rewrite, so prefer members whose rigid body lands at the same
  relative path.
