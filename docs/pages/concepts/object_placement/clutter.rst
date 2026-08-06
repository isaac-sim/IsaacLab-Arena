Clutter Placement
=================

Some arrangements cannot be solved for. A pile is defined by objects touching and resting on
one another, but ``On`` requires each child's whole footprint on a single Z plane and the
solver's global no-overlap term forbids contact outright. ``ClutteredOn`` instead drops objects
onto a support and lets physics decide where they come to rest.

Members are declared against a support and grouped by name. Unlike the relations in
:doc:`./relations`, they are not positioned by the solver: they are released above the support
and allowed to fall, and where they come to rest is the placement.

.. code-block:: python

   table.add_relation(IsAnchor())
   for tool in (mug, cracker_box, power_drill):
       tool.add_relation(ClutteredOn(table, group="tools", spread=0.8))

The equivalent YAML. Each member declares its own relation, and the ones naming the same
``reference`` and ``group`` are poured together as a single pile:

.. code-block:: yaml

   relations:
     - kind: cluttered_on
       subject: mug
       reference: table
       params: {group: tools, spread: 0.8}
     - kind: cluttered_on
       subject: cracker_box
       reference: table
       params: {group: tools, spread: 0.8}
     - kind: cluttered_on
       subject: power_drill
       reference: table
       params: {group: tools, spread: 0.8}

A support may hold several groups, and each is poured separately. Note that ``spread``
repeats identically across the three: it describes the region they share, so members that
disagree on it are rejected.

Parameters describing the pile, which every member must declare identically:

- ``group`` (default ``"clutter"``): ties members of one pile together
- ``spread`` (default ``1.0``): fraction of the support footprint to use, shrunk about
  its centre; must be in ``(0, 1]``. A pile is poured into one region, so members
  disagreeing on it is rejected.
- ``drop_order`` (default ``as_listed``): which members end up underneath; see below

Parameters describing a single member, which may differ between them:

- ``gap_m`` (default ``0.03``): vertical gap this object leaves above whatever it is
  released over
- ``clearance_m`` (default ``0.01``): height above the surface at which this object starts
- ``random_yaw`` (default ``True``): sample a yaw for this object before dropping. Turn it
  off for one member to drop it axis-aligned while the rest are turned.

Only the yaw is sampled. Any other release orientation is authored, by giving the member a
``RotateAroundSolution`` marker that the sampled yaw is then composed onto:

.. code-block:: python

    mug.add_relation(ClutteredOn(table, group="tools"))
    mug.add_relation(RotateAroundSolution(roll_rad=math.pi))  # released upside down

The drop planner refits the member's bounding box to that rotation, so a flipped object still
starts with its true lowest point at ``clearance_m`` above the surface. What cannot be expressed
is a *distribution* over orientations: every member carrying a marker is released at exactly that
rotation, in every environment and on every reset, and only its yaw varies. The marker is per
member, so half a pile can be marked upside down and half left alone; what cannot be expressed is
sampling that split per layout.

Resting orientations are another matter: the pile tumbles as it settles, so members come to rest
at whatever roll and pitch physics gives them, which is why a layout records full quaternions
rather than a yaw.

``drop_order`` decides the sequence members are planned in, which is how you choose which of
them end up underneath: each member is stacked above whatever earlier ones its footprint
overlaps:

- ``as_listed`` (default): the order the members appear in the asset list, so listing an object
  first puts it at the bottom
- ``flattest_first``: shortest first, so flat objects reach the surface before it gets lumpy
  and lie flat rather than coming to rest on an edge
- ``shuffle``: randomised per layout, so no one member sits at the bottom of every pile

It is pile-wide, so every member of a group must declare the same value. And it decides the
planned stack only: once physics runs, a pile landing on the first object can still shove it
aside, so it is a strong influence rather than a guarantee.

Not yet reachable from ``cluttered_on``, and worth knowing before designing around it:

- **No pile offset.** A group's region is always concentric with its support, so two groups on
  one support are poured into the same rectangle. They avoid each other's footprints, but
  "one pile on the left, one on the right" cannot be expressed; use two supports instead.
- **Orientation beyond yaw is authored, not sampled**, as above.

Because a pile is produced by simulation rather than optimisation, it differs from the
other relations in ways worth knowing:

- **The support must be provably immovable.** Its spawner must set
  ``rigid_props.kinematic_enabled``. ``IsAnchor`` is not enough, because it only fixes an
  asset for the solver's arithmetic and says nothing about simulation. An absent
  ``rigid_props`` is not enough either, since a background asset can spawn a rigid body through
  ``spawn_cfg_addon``.
  A pile is captured relative to where its support stood while settling, so a support that
  physics can shove replays the pile against a surface that has moved out from under it.
  This is conservative for a solver-placed rather than anchored support. Such a support is
  itself captured after settling and rewritten at reset alongside the pile, so the two stay
  consistent even if the pile shoves it. Relaxing the rule for that case would allow pouring
  into a movable bin, but it is untested and refused for now.
- **The support must be square to the world axes.** A rotation off a quarter turn is refused:
  the pour region and the region a settled pile is judged against are both axis-aligned boxes,
  so neither would describe a turned support's surface.
- **Nothing may be related to a member, and a member may declare little.** A relation naming a
  member as its parent is refused, and a member may carry only ``cluttered_on`` and a single
  ``RotateAroundSolution``. Members are held out of the solve, so anything else is discarded,
  and the two cases failed differently: naming a member as a parent reached the solver and
  raised a bare ``KeyError`` about a missing index, while a relation carried by a member was
  dropped without a word.
- **A support the layout does not place must declare a concrete Pose.** A ``PoseRange`` or
  ``PosePerEnv`` offers no single surface to pour onto. A solver-placed support may declare
  either, since its pose comes from the layout rather than the declaration.
- **The support must have a readable USD.** The drop region is derived from its bounding
  box, so a procedurally spawned support without geometry cannot be used.
- **A placement seed is required.** Clutter refuses to pour without one, since a pile that
  cannot be reproduced defeats the purpose of seeding a layout.
- **Layouts must resolve on reset.** A pile is settled after the environment is built (the
  default; ``settle_clutter_on_build`` turns it off for callers that settle themselves) and its
  resting poses are written back into the pool each reset draws from. Static placement keeps
  no such pool, so the combination is rejected rather than spawning a pile in mid-air.
- **Members never reach the solver or the build-time checks.** A pour assigns their poses,
  so any pose the optimiser gave them would be discarded; leaving them in would make them
  phantom obstacles that push the genuinely constrained objects around. They are therefore
  held out of the solver and out of the generic build-time validators, including overlap. The
  clutter-specific checks still run, including what a member may declare, whether the pile
  settled, and whether it stayed on its support.
- **A pile that always spills is a configuration error.** Layouts are settled once and the ones
  whose members left the support are discarded. An environment left with none fails at build
  time rather than running episodes against a pile lying beside its support: reduce the members,
  lower the group's ``spread``, or raise ``clutter_containment_margin_m``.
- **The pour avoids what is already on the surface instead.** Objects the solver placed on
  the same support, and members of earlier groups, are treated as occupied footprints and
  clutter is released above them rather than into them. A settled pile is in contact by
  definition, so checking it for overlap afterwards would reject correct arrangements.

Next Steps
----------

Continue to :doc:`./collision_handling` to learn how Arena checks placed assets against one
another and against fixed geometry.
