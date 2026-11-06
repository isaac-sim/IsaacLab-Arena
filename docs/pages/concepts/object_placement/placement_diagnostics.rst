Placement Diagnostics
=====================

The object placer converts an environment's relations into concrete object
poses. It solves batches of candidate layouts, then runs the
:doc:`pre-physics checks <./validation>` on each candidate: its relations,
collision clearance (``no_overlap``) and, when enabled, IK reachability
(``ik_reachable``). Placement diagnostics record how those checks went and why
they rejected candidates. Read them before changing relations.

What Is Recorded
----------------

Each fill of the :doc:`placement pool <./pooled_placement>`, at creation or
during a reset that finds a partition empty, records:

- for each solved batch, how many candidates passed every required check;
- for each check on that batch, how many candidates it ran on and how many
  passed, and every rejection it made (see below for what a rejection holds);
- which environments fell back to best-loss layouts.

A rejection has:

- a cause, naming the object, relation and what failed, such as
  ``'plate' next_to(box, positive_x): on the wrong side``. It holds no measured
  values, so rejections of the same kind are counted together;
- a measured violation, such as ``by 0.250 m``, saying by how much the
  candidate failed.

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

When a fill falls back to best-loss layouts, it also prints a summary of the
whole fill below the fallback message: the counts of each batch, then each
check's most frequent rejection causes over all batches, in the format above.
When a fill fails, it raises a ``PlacementPoolFillError`` whose message holds
the same summary and whose ``diagnostics`` hold the fill's record.

In an Experiment Run
--------------------

Each Run of an :doc:`Arena Experiment <../concept_arena_experiments>` writes
``arena_run_placement_diagnostics.json`` to its Run directory:

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
cause that rejects most candidates in every batch points to a problem in the
scene; more ``max_placement_attempts`` will not fix it. A cause that rejects a
few candidates now and then is usually harmless while the pool still fills with
valid layouts.

Common reasons for frequent rejections, which the measured violations help
tell apart, are:

- relations that conflict, such as two relations pulling an object in different
  directions;
- tolerances or margins too tight for the solver to meet;
- too little room, such as a support too small for the objects placed on it;
- bounds that do not match an object's shape, such as a bounding box that
  overestimates a concave object (see :doc:`./collision_handling`).

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
