Relations and Strategies
========================

Relations describe where a placeable asset should be positioned or oriented.
Attach them to an asset with ``add_relation()``. Arena considers all relations
on that asset together.

Positional relations use solver strategies that convert the requested
arrangement into optimization objectives. Orientation relations and placement
modifiers are handled separately. Most users keep the default strategies;
advanced users can replace entries in ``RelationSolverParams.strategies``.

.. code-block:: python

   from isaaclab_arena.relations.relations import IsAnchor, NextTo, On

   table.add_relation(IsAnchor())
   mug.add_relation(On(table))
   bowl.add_relation(On(table))
   bowl.add_relation(NextTo(mug))

This describes the intended arrangement without requiring coordinates derived
from the dimensions of the table, mug, and bowl.

For every relation's parameters, validation check and restrictions, see :doc:`/pages/api_reference/relations`.

.. _relations-anchors:

Anchors
-------

An anchor is a fixed reference in the relation graph. Mark it with
``IsAnchor()``; the solver does not move it. This marker does not make an asset
static or kinematic in physics. A standalone anchor needs a fixed
initial pose; in YAML, an omitted pose defaults to identity. An
``ObjectReference`` instead derives its pose from the referenced prim within
its parent asset. A tabletop or counter reference is a common anchor.

An anchor's fixed root rotation must be a multiple of 90 degrees about world Z,
with no tilt. For an ``ObjectReference``, this restriction applies to its parent
asset's pose; the referenced prim's authored rotation is already included in its
bounds. The solver rotates these bounds into world-aligned bounds.

When the support surface is part of a larger background, use an
``ObjectReference`` to identify that surface:

.. code-block:: python

   from isaaclab_arena.assets.object_reference import ObjectReference
   from isaaclab_arena.assets.object_type import ObjectType

   table_reference = ObjectReference(
       name="table",
       prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
       parent_asset=background,
       object_type=ObjectType.RIGID,
   )
   table_reference.add_relation(IsAnchor())
   mug.add_relation(On(table_reference))

Anchor the background asset directly when its complete bounds represent the
support. Use an ``ObjectReference`` when only an internal tabletop, counter, or
similar prim should support placement.

A background with mesh collision cannot have relations, ``IsAnchor`` included.
Placement would treat its whole mesh as one solid, which for a room fills the
open space, so placement fails with an error. Leave such a background without
relations, so it acts as a passive obstacle. To place objects on a surface
within the background, such as a countertop prim in a kitchen, anchor an
``ObjectReference`` to that prim, as above; see :doc:`./collision_handling`.

Combining Relations
-------------------

Relations are most useful in small combinations:

- ``On`` alone means "somewhere on this surface."
- ``On`` with ``NextTo`` means "on this surface, beside that object."
- ``On`` with ``AtPosition`` means "at this horizontal location on the
  surface."
- A positional relation with ``FaceTo`` controls both location and
  orientation.

Use ``NextTo`` for a side and an edge-to-edge clearance ("stand 20 cm in front
of the microwave"). Use ``AlignWith`` for a centre coordinate or a relative
offset ("line up with the microwave centre"). They can be combined: ``NextTo``
sets the front clearance and ``AlignWith`` fixes lateral alignment. Two
``AlignWith`` relations instead define a centre-relative X/Y target region;
that does not preserve an edge clearance when either object's size changes.

Avoid specifying more relations than the environment needs. Extra constraints
can make the intended layout harder or impossible to satisfy.

Placement Modifiers
-------------------

``RotateAroundSolution`` is a pose modifier applied after solving. It changes
how a solved pose is used rather than adding a spatial constraint: it adds a
fixed roll, pitch, or yaw to the solved pose. The robot-placement example later
in this sequence uses it to set the robot's final heading.

To randomize an object's position within a region, constrain it with
``PositionLimitsBox`` instead of pinning it with ``AtPosition``. Each solved
layout lands somewhere inside the box, so resets drawn from the layout pool
spread the object across it.

Relations in Environment Specifications
---------------------------------------

YAML environment specifications use the same model:

.. code-block:: yaml

   relations:
     - kind: is_anchor
       subject: table
     - kind: 'on'
       subject: mug
       reference: table
     - kind: next_to
       subject: bowl
       reference: mug

Each entry identifies the relation, its subject, and—when needed—the object it
references. Add parameters only when the default relation does not express the
intended arrangement. Quote ``'on'`` so YAML treats it as a string rather than
a Boolean value.

Collision handling is integrated into placement and is not expressed as a
relation.

.. _recorded-layouts:

Replay a Recorded Layout
-------------------------

Placement poses can be replayed from an episode JSONL containing solved poses
from earlier runs, bypassing relation solving and post-physics settling.

Following is an example Experiment Definition with an episode JSONL for replay:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/placement_record_replay/settled_placement_replay_experiment.yaml
   :language: yaml

Run it from the repository root:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
       --experiment_config isaaclab_arena_environments/experiment_configs/placement_record_replay/settled_placement_replay_experiment.yaml \
       --device cpu --viz kit

Python callers use
``ArenaEnvBuilderCfg(recorded_variation_samples_path="layouts.jsonl")``. All
file paths are relative to the working directory.
Placement replay does not rerun geometry, reachability or settling checks.
``resolve_on_reset`` and ``placement_seed`` are ignored.

File Format
~~~~~~~~~~~

Each row contains one complete sample under ``placement``:

.. code-block:: json

   {"placement": {"layout_id": "layout_000000", "poses": {"cube": {"position_xyz": [0.1, 0.2, 0.3], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}}}}

Pose keys are runtime scene-root names. Ordinary objects use their instance
names; a single-root embodiment commonly uses ``"robot"``; compound assets
must include every name returned by ``asset.get_scene_root_keys()``. Positions
are environment-local metres and rotations are xyzw quaternions.

.. _placement-replay-configuration:

Replay Configuration
~~~~~~~~~~~~~~~~~~~~

Replay validates finite poses, unit quaternions, consistent root coverage, and
reset ownership before constructing the scene. It rejects:

- unknown roots or incomplete compound assets;
- missing non-anchor assets constrained by relations;
- ``RigidObjectSet`` assets, whose identity cannot be recovered from poses
  alone;
- disabled pose resets, nonzero initial root velocity, or non-fixed
  asset-owned pose-reset policies.

After validation, one coordinated placement event becomes the root-pose reset
writer for all recorded assets and zeros their root velocities. Articulation
joint reset behavior remains independent.


Next Steps
----------

See :doc:`../offline_placement/recording` to settle layouts and save poses
for reuse. For table and container examples, see
:doc:`../offline_placement/clutter`.

Continue to :doc:`./collision_handling` to learn how Arena checks placed assets
against one another and against fixed geometry.
