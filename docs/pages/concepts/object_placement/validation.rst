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

1. **Inexpensive checks** (``no_overlap``, ``on_relation``, ``clutter_on_relation``, ``next_to``,
   ``not_next_to``, ``face_to``) over every candidate.
2. **Expensive checks** (``ik_reachable``) only on candidates that already
   passed every *required* inexpensive check.

A ``PlacementCandidate`` is one working layout of all placement objects in one
environment. It stores the objects' positions, orientations and bounds, plus the
environment ID and sample ID. Solving updates that same candidate's positions and
loss; validation replaces its check results. These stages update the working batch
in place and return ``None``.

``PlacementCandidateBatch`` groups these layouts. For example, four environments
with ten attempts each produce a batch of forty candidates. Filtering and ranking
select and reorder references to complete candidates, keeping their geometry and
results together.

``PlacementCandidateGenerator`` samples orientations, fits the bounds to those
rotations and places clutter above its support and nearby objects before solving.
After solving, ``ObjectPlacer`` applies ``FaceTo`` headings and refits the bounds.
``PlacementValidationRunner`` then replaces each candidate's check results.
Here, *expensive* means computational cost, such as solving IK. Neither validation
pass steps physics.

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
   * - ``clutter_on_relation``
     - Build-time
     - Inexpensive
     - Yes
     - Every ``ClutterOn`` release fits its region and clears its support.
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

``no_overlap``, ``on_relation``, ``clutter_on_relation``, ``next_to``,
``not_next_to``, and ``face_to`` are always registered. The first five mirror the corresponding
relation's loss term, so low-loss solver output and validator verdicts stay
consistent. ``face_to`` is different: ``FaceTo`` is applied as a post-solve
heading rather than a continuous loss, and the check only verifies that a
facing yaw was computed. A check for a relation kind unused in the
environment passes trivially.

Set ``debug_visualize=True`` (or ``placer_params.debug_visualize: true`` in
YAML) to inspect candidates in a `Rerun <https://rerun.io/>`_
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

Both are set on ``ObjectPlacerParams`` in Python or the ``placer_params`` block
in YAML; see :doc:`../environment/environment_definition`.

Post-Physics Checks for Recordings
----------------------------------

The :doc:`placement recorder <../offline_placement/recording>` advances physics
for ``settle.num_steps`` environment steps before checking the final state.

All required solver checks and enabled, applicable post-physics checks must pass.
The post-physics checks share one settling pass:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Check
     - Measures
     - Default limit
   * - ``physics_settled``
     - Final linear and angular speed of every root, including the robot.
     - 0.1 m/s and 0.1 rad/s
   * - ``pose_shift``
     - Initial-to-final translation and rotation of every root.
     - 2 mm and 2 degrees
   * - ``articulation_link_shift``
     - Task-object link motion relative to its root, excluding robot embodiments.
     - 2 mm and 2 degrees

Configure ``settle.validators.<check>.<setting>`` to change a limit.
An inapplicable or disabled check is recorded as skipped. At least one applicable
post-physics check must remain enabled.

The recorder writes a file only when at least ``settle.min_layouts`` layouts
are accepted across all batches (default 1). Otherwise, it exits with an error
and writes no file.

.. _recording_rejection_summary:

Understand the Rejection Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``Rejected`` line gives a reason and the number of layouts rejected for it.
When a pose-shift limit is exceeded, the message includes the object, measured
translation and rotation, and configured limits. Use the named
check to decide what to adjust:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Rejection
     - Meaning
     - What to check or change
   * - ``solver validation failed``
     - A layout failed a required solver check.
     - Read earlier solver diagnostics and check relation constraints and collision
       geometry. Increasing settling time cannot fix a solver failure.
   * - ``physics_settled``
     - Final root velocity is too high.
     - Inspect contacts and support. Increase ``settle.num_steps`` if motion is
       still decaying. More time will not fix an unstable placement.
   * - ``pose_shift``
     - A root moved or rotated beyond its limits.
     - Inspect the named object. For ``On`` relations, reduce ``clearance_m`` to
       reduce the release drop, or increase ``edge_margin_m`` to avoid support
       edges. Edit the relation's ``params`` in the environment's scene YAML.
   * - ``articulation_link_shift``
     - A task-object link shifted relative to its root.
     - Check joint initialization and contacts. A root-pose recording cannot
       preserve the changed joint configuration.
   * - ``missing required solver checks``
     - A required pre-physics result is unavailable.
     - Enable the required validator or fix its dependencies and source
       configuration.

See :doc:`./relations` for relation settings. Change validation
limits under ``settle.validators`` only when the new tolerance fits your evaluation.
Increasing ``layouts_per_env`` samples more candidates without changing which
checks they must pass.

Custom Validators
-----------------

Build-time checks subclass ``PrePhysicsPlacementValidator`` from
``isaaclab_arena.relations.validation.pre_physics`` and register with
``register_validator`` from ``isaaclab_arena.relations.validation.registry``.
Implement ``validate_batch(batch, collision_objects)`` to return one boolean per
candidate and give the class a unique ``check`` name. Include that name in
``ObjectPlacerParams.enabled_checks`` when explicitly selecting checks.
Existing extensions importing the build-time ``PlacementValidator`` should use
``PrePhysicsPlacementValidator`` instead. The constructor is unchanged, but
``validate_batch`` now takes a ``PlacementCandidateBatch`` instead of separate
position, orientation and bounding-box lists. Import
``PlacementCheck``, ``PlacementValidationResults`` and ``PlacementValidatorReport``
from ``isaaclab_arena.relations.validation.types``.

For example, reject layouts with any object origin above 1.5 metres:

.. code-block:: python

   from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
   from isaaclab_arena.relations.validation.registry import register_validator

   @register_validator
   class MaxOriginHeightValidator(PrePhysicsPlacementValidator):
       check = "max_origin_height"

       def validate_batch(self, batch, collision_objects):
           verdicts = []
           for candidate in batch.candidates:
               below_limit = all(
                   position[2] <= 1.5 for position in candidate.positions.values()
               )
               verdicts.append(below_limit)
           return verdicts

Import the module defining this class before constructing ``ObjectPlacer``.
Default settings run and require the registered check. If ``enabled_checks`` is
explicit, include ``"max_origin_height"`` to run it.

The shared ``PlacementValidator`` in ``isaaclab_arena.relations.validation.base``
defines the check name and stage. Post-physics checks use
``PostPhysicsPlacementValidator`` from
``isaaclab_arena.offline_placement.post_physics_validation``.

Next Steps
----------

Continue to :doc:`./pooled_placement` for how Arena ranks, stores, and reuses
layouts that pass these checks.
