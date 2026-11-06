Placement Diagnostics
=====================

The object placer converts an environment's relations into concrete object
poses. It solves batches of candidate layouts, then runs the
:doc:`pre-physics checks <./validation>` on each candidate: its relations,
collision clearance (``no_overlap``) and, when enabled, IK reachability
(``ik_reachable``). Placement diagnostics record how those checks went and why
they rejected candidates. Read them before changing relations or writing a
diagnostic script.

What Is Recorded
----------------

Each fill of the :doc:`placement pool <./pooled_placement>`, at creation or
during a reset that finds a partition empty, records:

- for each solved batch, how many candidates passed every required check;
- for each check on that batch, how many candidates it ran on and how many
  passed, and every rejection it made;
- which environments fell back to best-loss layouts.

A rejection has a cause and a measured violation. The cause names the object,
relation and what failed, such as
``'plate' next_to(box, positive_x): on the wrong side``. It holds no measured
values, so rejections of the same kind are counted together. The measured
violation, such as ``by 0.250 m``, says by how much the candidate failed.

On the Console
--------------

After each batch, Arena prints how many candidates passed each check, then one
line per rejection cause with its count and one measured violation. For the
KitchenBench ``soup_in_far_bowl`` environment, for example:

.. code-block:: text

   [placement] Validated 1500 candidate layout(s); passed per check: on_relation=1304/1500, ..., no_overlap=1484/1500
     on_relation: 'far_bowl' on(island_top): bottom outside the contact band x192 (e.g. bottom z 0.973 m, support top z 0.880 m)
     on_relation: 'near_bowl' on(island_top): bottom outside the contact band x2 (e.g. bottom z 0.895 m, support top z 0.880 m)
     align_with: 'far_bowl' align_with(near_bowl, x): centre outside tolerance x1 (e.g. centre error +0.069 m, tolerance 0.06 m)
     no_overlap: 'far_bowl' and 'fixed_collision_mesh': meshes closer than the clearance x16 (e.g. gap short of the 0.010 m clearance by up to 0.001 m)

With ``ObjectPlacerParams.verbose`` set, each check also prints its rejections
after it runs on a batch, grouped by cause and at most
``max_printed_rejections_per_cause`` (3 by default) for each cause; one more
line counts the rest. In an environment spec YAML, set ``verbose: true`` under
``placer_params``.

When a fill falls back to best-loss layouts, it prints a summary below the
fallback message: one line per batch with its counts, then each check's most
frequent rejection causes over the fill.

When a fill fails, it raises a ``PlacementPoolFillError`` whose ``diagnostics``
hold the fill's record and whose message holds the same summary. For a box
wider than the table it is placed on, for example:

.. code-block:: text

   batch 0: 0/4 valid; passed per check: on_relation=0/4, clutter_on_relation=4/4, ..., no_overlap=4/4
   rejections by cause over all batches:
     on_relation: 'box' on(desk): edge margin wider than the free space x8

From Python
-----------

- After ``build_env_cfg()`` or ``make_registered()``,
  ``ArenaEnvBuilder.placement_diagnostics`` holds the record of the fill made
  during that build. It is ``None`` when the build solved no relations or
  failed.
- From a running environment, ``get_placement_diagnostics(env)`` returns the
  record of the pool's most recent fill, including one made during a reset, or
  ``None`` when the environment has no live pool. It is in
  ``isaaclab_arena.relations.runtime.placement_events``.
- A record's ``format_summary()`` returns the summary printed on fallback, and
  ``get_num_rejected_by_cause_per_check()`` returns each check's rejections
  counted by cause, most frequent first.

In an Experiment Run
--------------------

Each Run of an :doc:`Arena Experiment <../concept_arena_experiments>` writes
``arena_run_placement_diagnostics.json`` to its Run directory, whatever the
Run's policy:

- The file holds one entry per build that solved relation placement, with
  everything a fill records, including every rejection.
- A build that replays recorded layouts solves nothing, so it adds no entry.
- A failed Run still writes the file, with the builds it made and, when a fill
  failed, that fill.

``AggregatedPlacementDiagnostics.load()`` in
``isaaclab_arena.relations.placement_diagnostics`` reads the file. The
:doc:`./placement_success_benchmark` computes its metrics from these files.

Acting on Rejections
--------------------

Start with the required check that rejects the most candidates and its most
frequent cause. A check that runs only on candidates that passed the cheaper
required checks, such as ``ik_reachable``, counts only those candidates. A
cause that rejects most candidates in every batch points to relations that
conflict or a support that is too small; more ``max_placement_attempts`` will
not fix it. A cause that rejects a few candidates now and then is usually
harmless while the pool still fills with valid layouts.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Check
     - Cause
     - What to check or change
   * - ``on_relation``, ``clutter_on_relation``
     - bottom outside the contact band
     - The child's bottom is not within the relation's ``clearance_m`` above
       the support top, give or take ``on_relation_z_tolerance_m``. Compare the
       two heights in the measured violation. Check that the support's bounding
       box top is its real surface, and that nothing else on the support lifts
       the child.
   * -
     - edge margin wider than the free space
     - The support cannot fit the child with ``edge_margin_m`` on every side.
       Reduce ``edge_margin_m``, or use a larger support.
   * -
     - footprint outside the support
     - The child overhangs the support. Give it more room, with fewer or smaller
       objects on the support or a larger support, or set ``overlap=True`` to
       allow partial overlap.
   * - ``align_with``
     - centre outside tolerance
     - Raise ``tolerance_m`` or correct ``offset_m``. Check that another
       relation, such as ``NextTo``, does not pull the child off the line.
   * - ``next_to``
     - on the wrong side
     - The child ended up on the other side of the reference. Check for
       relations that pull it back and for room on the requested side.
   * -
     - gap outside tolerance
     - The gap misses ``distance_m`` by more than ``tolerance_m``. Raise
       ``tolerance_m``, or check that the support has room at that distance.
   * - ``not_next_to``
     - inside the keep-out zone
     - The child could not leave the zone beside the reference. The measured
       violation gives the distance still needed on each way out. Give it room
       elsewhere on the support.
   * - ``face_to``
     - too close to its target in XY to define a heading
     - The subject and its target nearly coincide in XY. Separate them, for
       example with a ``NextTo`` relation.
   * - ``no_overlap``
     - bounding boxes closer than the clearance
     - Two boxes are closer than ``RelationSolverParams.clearance_m``. Give the
       objects more room. Boxes overestimate concave or rotated objects; mesh
       collision is tighter, see :doc:`./collision_handling`.
   * -
     - meshes closer than the clearance
     - Two collision meshes are closer than ``clearance_m``. Give the objects
       more room, or reduce ``clearance_m``.
   * -
     - SDF query found no mesh face
     - The collision query failed on a mesh. Check the object's collision mesh.
   * - ``ik_reachable``
     - top-down grasp unreachable
     - The robot cannot reach a top-down grasp on the object within
       ``ReachabilityConfig``'s IK thresholds. Place the object closer to the
       robot, or move the robot.
