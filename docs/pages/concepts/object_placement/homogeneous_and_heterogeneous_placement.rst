Homogeneous and Heterogeneous Object Placement
==============================================

These terms describe object identity across parallel environments, not whether
the layouts are identical:

- **Homogeneous placement** uses the same registered object for a role in every
  environment. Its solved pose can still differ between environments and
  resets.
- **Heterogeneous placement** uses a
  :doc:`PerEnvironmentObject <../scene/concept_object_variants>` to assign one
  object to each environment, so object geometry can differ across environments.

Both modes use the same relations, solver, validators, and layout-pool
workflow. They differ in how Arena chooses objects and supplies their geometry
to the solver.

Implementation Differences
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Stage
     - Homogeneous objects
     - Heterogeneous objects
   * - Environment definition
     - Add a registered ``Object`` directly.
     - Add a ``PerEnvironmentObject`` listing the objects that can fill the role.
   * - Per-environment assignment
     - The environment clones one ``Object`` definition, so the same USD fills
       the role in every environment.
     - Arena selects one variant for each environment at build time, after
       the environment count is known and before assets are spawned.
       ``assign_to_environments="sequential"`` repeats the declared order;
       ``"random"`` samples independently.
   * - Dimensions used by spatial relations
     - Arena broadcasts the object's bounding box to every environment.
     - Arena uses the selected variant's bounding box in each environment.
   * - Construction
     - Every environment spawns the same USD for that role.
     - Every environment spawns the USD selected for that environment.
   * - Reset
     - The object identity stays fixed while the layout may change.
     - The selected variant stays fixed while the layout may change.

These bounding boxes provide object dimensions for spatial relation solving.
Collision checks separately use the configured ``BBOX`` or ``MESH``
representation.

Setting ``placement_seed`` makes random object variant assignments and layout
generation reproducible. Arena fixes object variant assignments while building the
environment and keeps them unchanged for its lifetime so that spawned USDs and
the geometry used for placement remain aligned.

Homogeneous Example
-------------------

The maintained ``droid_table_multi_object_placement`` environment places the
same five registered objects on a Maple table in every environment in
homogeneous mode. The solver can produce a different layout for each
environment and reset.

.. figure:: ../../../images/same_objects_different_layouts.gif
   :width: 100%
   :alt: The same objects placed in different layouts across four environments
   :align: center

Run the same registered environment configuration shown in the animation:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_envs 4 \
     --env_spacing 3.0 \
     --placement_seed 42 \
     --resolve_on_reset \
     --num_steps 500 \
     droid_table_multi_object_placement \
     --embodiment droid_abs_joint_pos \
     --episode_length_s 4.0 \
     --mode homogeneous

Heterogeneous Example
---------------------

The same registered environment uses five ``PerEnvironmentObject`` entries in
heterogeneous mode: a fruit, bottle, can, tool, and box for each environment.
The solver uses every selected variant's dimensions.

.. figure:: ../../../images/heterogeneous_placement.gif
   :width: 100%
   :alt: Different object variants placed across four parallel environments
   :align: center

The environment creates each heterogeneous role from registered variants and
attaches the same ``On`` relation used for homogeneous objects. Its builder
uses:

.. code-block:: python

   from isaaclab_arena.assets.per_environment_object import PerEnvironmentObject

   for role_name, variant_names in HETEROGENEOUS_VARIANT_SETS.items():
       assets = self._build_registered_objects(variant_names)
       obj = PerEnvironmentObject(name=role_name, objects=assets)
       obj.add_relation(On(table_reference))
       placeable_assets.append(obj)

Run the heterogeneous configuration shown in the animation:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_envs 4 \
     --env_spacing 3.0 \
     --placement_seed 42 \
     --resolve_on_reset \
     --num_steps 500 \
     droid_table_multi_object_placement \
     --embodiment droid_abs_joint_pos \
     --episode_length_s 4.0 \
     --mode heterogeneous

The builder must know ``num_envs`` before assigning object variants. The
runner passes this count through ``--num_envs``; use a value greater than one to
observe different variants across parallel environments.

.. important::

   Do not set an initial pose on an object whose pose is determined by
   placement relations. The builder supplies its creation and reset poses.
   Anchors remain fixed and therefore still need a known pose.

Related References
------------------

Refer to :doc:`../scene/concept_object_variants` for native variant
configuration, :doc:`./pooled_placement` for reset behavior and object
assignment, or :doc:`./relations` for the available spatial relations.
