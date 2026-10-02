Placement Validation
====================

The solver (:doc:`./solver`) minimizes a continuous loss over relations and
collisions. A low loss does not guarantee that a relation holds exactly, that
a robot can reach a target, or that a property the loss never encoded is
satisfied. Validators are the pass/fail layer on top of that output: each
re-checks one property of a solved candidate and reports ``True``/``False``,
independent of the loss value.

The solver asks how to reduce violations; a validator asks whether a specific
candidate actually satisfies the property it checks. Validation is also where
constraints the solver never optimizes enter the pipeline, for example IK reachability.

How Validation Fits Placement
-----------------------------

``ObjectPlacer`` builds its validator list once from every registered check
that passes ``is_available()`` and survives ``enabled_checks`` (see
:ref:`validation-toggle`). Each solved batch then runs in two passes:

1. **Inexpensive checks** (``no_overlap``, ``on_relation``, ``next_to``,
   ``not_next_to``, ``face_to``) over every candidate.
2. **Expensive checks** (``ik_reachable``) only on candidates that already
   passed every *required* inexpensive check.

Verdicts land in each candidate's ``PlacementValidationResults``. A
**required** check must pass for the candidate to count as valid; an
**optional** (enabled but not required) check still runs and is reported, but
does not invalidate the layout. ``PooledObjectPlacer``
(:doc:`./pooled_placement`) uses these results to rank candidates and to
reject-and-refill until each environment has enough valid layouts. By default,
if the final refill batch still has no valid candidate, Arena can store a
best-loss layout that failed required checks
(``allow_best_loss_fallbacks=True``); see :doc:`./pooled_placement`.

Types of Validators
--------------------

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 18 40

   * - Check (``PlacementCheck``)
     - Stage
     - Cost
     - Enabled by default
     - What it checks
   * - ``no_overlap``
     - Build-time
     - Inexpensive
     - Yes
     - No two placed bounding boxes (or collision meshes in ``MESH`` mode)
       intersect. See :doc:`./collision_handling`.
   * - ``on_relation``
     - Build-time
     - Inexpensive
     - Yes
     - Every ``On`` relation holds (XY footprint and Z band).
   * - ``next_to``
     - Build-time
     - Inexpensive
     - Yes
     - Every ``NextTo`` relation holds within tolerance.
   * - ``not_next_to``
     - Build-time
     - Inexpensive
     - Yes
     - Every ``NotNextTo`` keep-out zone is cleared within tolerance.
   * - ``face_to``
     - Build-time
     - Inexpensive
     - Yes
     - Every ``FaceTo`` subject has a well-defined facing yaw.
   * - ``ik_reachable``
     - Build-time
     - Expensive
     - When available
     - cuRobo IK can reach a top-down grasp at every
       ``RequiresReachability`` object.

Geometric and Relation Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``no_overlap``, ``on_relation``, ``next_to``, ``not_next_to``, and
``face_to`` are always registered. The first four mirror the corresponding
relation's loss term, so low-loss solver output and validator verdicts stay
consistent. ``face_to`` is different: ``FaceTo`` is applied as a post-solve
heading rather than a continuous loss, and the check only verifies that a
facing yaw was computed. A check for a relation kind unused in the
environment passes trivially.

Set ``debug_visualize=True`` (or ``placement_validators.debug_visualize:
true`` in YAML) to inspect candidates in a `Rerun <https://rerun.io/>`_
viewer:

.. figure:: ../../../images/validator_bbox_rerun_viz.gif
   :width: 100%
   :alt: Rerun debug view of base placement validators with bounding boxes
      and a per-check pass/fail timeline.
   :align: center

   Base debug view: anchors in gray, movable objects in blue. The
   ``checks/`` stream plots per-candidate pass/fail for the inexpensive
   geometric checks.

.. _ik-reachable-check:

IK Reachability (``ik_reachable``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../../images/reachability_black_into_bin_bbox_viz.png
   :width: 100%
   :alt: Candidate layouts with reachable task objects in green, unreachable
      distractors in red, and a light-green reachability voxel field.
   :align: center

   ``ik_reachable`` gates only objects marked ``RequiresReachability``
   (green). Distractors outside the reachable volume may appear red for
   illustration, but they never reject the layout.

``ReachabilityValidator`` lives in the optional ``isaaclab_arena_curobo``
extension. It registers when that package imports successfully and the
embodiment has a cuRobo config (e.g. Droid). ``ArenaEnvBuilder`` wires
the env's embodiment into ``reachability_config`` automatically.

It gates only objects marked ``RequiresReachability`` — a relation that tasks
such as ``PickAndPlaceTask`` stamp automatically. With no such relation, the
check passes trivially. Grasp offset and IK tolerances are configurable; see
:doc:`../environment/environment_definition`.

.. figure:: ../../../images/reachability_rerun_viz.gif
   :width: 100%
   :alt: Rerun view of IK reachability with green and red robot collision
      spheres above target bounding boxes.
   :align: center

   With ``debug_visualize`` on, green spheres mark a reachable,
   collision-free grasp; red marks a failure that rejects the candidate.

.. _validation-toggle:

Enabling and Disabling Checks
-------------------------------

Two concepts control the build-time checks: which checks **run**
(``enabled_checks``) and, of those, which must **pass** for a layout to be
valid (``required_checks``). By default every registered check runs and is
required — so when cuRobo is available, ``ik_reachable`` gates placement
automatically. Making a check optional (enabled but not required) keeps it
running and reported without rejecting layouts, which is the usual way to
keep placement geometry-only.

Both are set on ``ObjectPlacerParams`` in Python or the
``placement_validators`` block in YAML; see :doc:`../environment/environment_definition`
for the field-level split.

Next Steps
----------

Continue to :doc:`./pooled_placement` for how Arena ranks, stores, and reuses
layouts that pass these checks.
