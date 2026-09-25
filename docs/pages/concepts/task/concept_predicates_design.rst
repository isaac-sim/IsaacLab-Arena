Predicate Library Runtime Design
================================

**Status:** Proposed

**Scope:** The run-time architecture used by geometry-aware predicates in
``IsaacLabArenaManagerBasedRLEnv``. The first implementation targets object
containment and the task-level stable-placement condition built from it.
The existing pick-and-place progress objective uses that same condition so its
final milestone agrees with task success, but changes to the progress-tracking
architecture and a general stateful-predicate framework are follow-up work.
Mimic is out of scope and has no compatibility path in this design because it
is slated for removal from Arena.

**Implementation baseline:** Arena ``main`` at ``cf3a825b`` and RoboLab at
``f5acaa51``. The later RoboLab instance-proxy traversal was separately checked
at ``ce784a60``. The design follows executed predicate code where RoboLab's
docstrings disagree with it.


Decision summary
----------------

Arena objects, object references, object sets, and scenes are pre-simulation
configuration. They participate in scene composition and placement, but they do
not hold or query a live simulator environment. Affordances are the deliberate
exception to this general pre-simulation boundary.

Run-time predicates operate on the live environment using scene-name strings:

.. code-block:: python

   def object_in_container(
       env,
       object_name: str,
       container_name: str,
   ) -> torch.Tensor:
       ...

The live environment provides the two kinds of data a spatial predicate needs:

* ``env.scene[name]`` remains the authoritative store of Isaac Lab live
  entities.
* ``env.arena_world`` is Arena's public, name-based run-time query facade. It
  reads mutable state from ``env.scene`` and owns geometry derived from the
  composed USD stage for the lifetime of that environment.

``ArenaWorld`` is not a second object registry and does not contain Arena
configuration objects. Its cache stores derived geometry and immutable
source-to-environment resolution metadata, never mutable simulation state.
There is no ``ArenaObjectRuntimeSpec``.

Spatial relationships remain atomic and reusable. A task-level placement
condition composes containment, upward support, and low velocity instead of
making contact or motion part of the meaning of “inside.”

``PickAndPlaceTask`` keeps Arena's existing shared ``on`` model for containers
and volumetric support surfaces. Its success and final progress milestone use
one canonical stable-placement callable, and the unused ``max_separation``
option is removed.


Why redesign the predicate library
----------------------------------

Arena's predicate package began as a small set of task-specific helpers. It is
now being asked to express spatial relationships that need both current
simulation state and geometry. The existing implementation does not provide a
clean boundary for that combination.

Inconsistent predicate contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current predicates use a mixture of scene-name strings and
``SceneEntityCfg`` values. Some unwrap Gym environments before accessing the
scene while others access ``env.scene`` directly. Several predicates optionally
select one environment and return a scalar, although Isaac Lab manager terms
and Arena progress tracking expect one Boolean per parallel environment.

This makes predicates harder to compose and obscures which arguments are
ordinary configuration and which identify live scene entities.

Task success does not express spatial placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On current ``main``, the default pick-and-place success term checks only that a
filtered contact force exceeds a magnitude threshold and that the object has
low linear velocity. It does not check that the object is geometrically inside
its destination, and force magnitude alone cannot distinguish upward support
from lateral or downward contact. An object touching the outside of a
destination can therefore satisfy the default condition.

An optional proximity term compares axis-aligned center separation, but it is
disabled by default and explicitly does not support an ``ObjectReference``
destination. The redesign needs an explicit spatial relationship whose
documented semantics are shared by task success and the final progress
milestone.

Pre-simulation objects access live simulation state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ObjectBase.get_object_pose`` and ``ObjectBase.set_object_pose`` accept a live
environment and reach into ``env.scene``. These methods exist on current
``main`` and are used by repository tests. They will be migrated to run-time
helpers and removed from ``ObjectBase``.

That direction conflicts with Arena's lifecycle boundary. Arena objects describe
what will be composed before simulation starts; they are not handles to what was
spawned afterwards.

No owner exists for run-time geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``env.scene`` already owns the live Isaac Lab entities. Arena's local bounding
boxes, however, are cached on pre-simulation object configurations for placement
and relation solving. Reusing those objects at run time crosses the lifecycle
boundary; querying USD geometry on every predicate call would instead put stage
traversal in the simulation hot path.

The missing abstraction is therefore a name-based run-time query facade with a
cache for geometry derived from the composed live scene, owned by the live
environment.

One scene name can have several geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``RigidObjectSet`` has one scene name, but a different member can be spawned
in each parallel environment. A cache that maps one name to one bounding box can
silently use the wrong geometry.

Object references introduce a related problem. Their geometry and pose frame
may be nested below a parent scene asset, and the parent itself may be
heterogeneous. Run-time resolution must use the clone plan and composed prims,
not the original parent configuration.

Geometry semantics and frames are implicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The obsolete PR #979 containment experiment combined a cached pre-simulation
bounding box with a live pose. Current ``main`` has no central contract defining
which prim owns a pose, which local frame contains derived geometry, whether
scale is already applied, or which USD representation a run-time geometry query
uses.

Computing a world-aligned bounding box and subtracting translation does not
produce a correct local box for rotated geometry. The geometry extractor must
express its result in the same frame whose live pose the predicate reads.

Cache lifecycle is undefined
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static local geometry should be computed once, survive episode resets, and be
discarded with its environment. Current poses must always be read afresh.

Run-time architecture is not covered by tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current tests exercise selected task outcomes but do not verify a geometry-aware
predicate against composed live USD geometry. The obsolete PR #979 tests mostly
replace the environment, Arena objects, and clone plan with test doubles. They
therefore do not prove source resolution, instance handling, pose-frame
alignment, heterogeneous variants, cache isolation, or teardown in a live
simulation.

Stateful predicates have separate lifecycle concerns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``objects_settled`` both evaluates a condition and records the first settled
pose. Progress tracking currently resets that recorder. Consequently, using the
predicate outside progress tracking can retain episode state unexpectedly.

This is a real library shortcoming, but it is distinct from environment-lifetime
geometry caching. A later design should give episode-lifetime predicate memory
an explicit owner without coupling it to progress tracking.


Lessons from RoboLab
--------------------

RoboLab's public atomic conditionals receive an environment and scene-name
strings. Their lower-level predicate functions receive a ``WorldState`` facade,
which resolves names through ``env.scene``, reads live poses on every call, and
lazily caches local AABBs and convex hulls. The implementation reviewed for
this design is RoboLab commit ``f5acaa51``, chiefly
``robolab/core/world/world_state.py`` and
``robolab/core/task/predicate_logic.py``.

The separation between fresh pose data and cached local geometry is useful.
Arena should not copy two other parts of the implementation:

* RoboLab reaches ``WorldState`` through a module-global singleton. Arena must
  support independently owned, simultaneously live environments.
* RoboLab caches geometry by body name after reading environment zero. This
  assumes every environment has identical geometry and does not support
  ``RigidObjectSet``.

RoboLab's cache is also split across a declared local-geometry dictionary and
hull dictionaries attached lazily by predicate code. Arena keeps these derived
values behind one owned facade with one lifecycle.

The implementation also provides useful reference semantics for the first
Arena predicates. RoboLab's ``object_in_container`` ultimately tests the
object's convex-hull centroid against the container's cached open-top convex
hull. Its docstring still describes an older AABB implementation, so this
design follows the executed code rather than that description. Optional flags
can additionally require contact, gripper detachment, or low linear velocity.

RoboLab's ``object_on_top`` combines two different atomic facts: the object's
centroid is within the support surface's AABB footprint, and the net contact
force supports the object upward within a 45-degree cone. Arena uses the same
separation of geometry, contact, and motion, but keeps the atomic predicates
explicit before composing them for a task.

RoboLab can find the required contact sensor from the object and surface names
because it creates pairwise sensors with deterministic names. Arena currently
creates task-specific filtered contact sensors instead. The first Arena
implementation therefore receives that sensor's scene name explicitly rather
than adding contact-sensor registration to this redesign.

RoboLab's composite ``pick_and_place`` condition additionally requires the
object to be detached from the gripper. Arena does not yet expose a uniform
named gripper-contact interface across embodiments, so gripper detachment is a
deliberate follow-up rather than part of the first stable-placement condition.


Lessons from the obsolete PR #979 approach
------------------------------------------

PR #979 extended pre-simulation Arena objects with live-state methods and
passed those objects through manager configuration. Because Isaac Lab
deep-copies manager configurations, the experiment added ``ArenaAssetHandle``
with identity-preserving copy behavior and per-term adapters. It also rebuilt
object-set geometry selection and device tensors in the predicate hot path.

These are useful failure modes, not code to preserve. They show why the new
implementation uses native string parameters, an environment-owned geometry
cache, and live integration tests. No part of PR #979 is an implementation base
for this design.


High-level design
-----------------

The run-time flow is:

.. code-block:: text

   task configuration
       -> predicate parameters contain scene-name strings and scalar settings
       -> predicate(env, object_name, container_name)
          `-> env.arena_world
              |-> env.scene[name]             current simulation state
              |-> env.scene[sensor_name]      current contact state
              `-> local geometry cache
                    | cache miss
                    `-> clone plan + composed source prims

``IsaacLabArenaManagerBasedRLEnv`` owns one public ``ArenaWorld`` for its entire
live lifetime. Predicate code receives the unwrapped environment from Isaac Lab
managers and accesses ``env.arena_world`` directly. Code starting from a
Gym-wrapped environment uses ``env.unwrapped.arena_world``.

The initial public surface is deliberately query-oriented:

.. code-block:: python

   env.arena_world.get_pose_w(object_name)                    # fresh
   env.arena_world.get_pose_env(object_name)                  # fresh
   env.arena_world.get_linear_velocity_w(object_name)         # fresh
   env.arena_world.get_local_hull_groups(object_name)         # cached
   env.arena_world.get_filtered_contact_force(sensor_name)    # fresh

``ArenaWorld`` delegates entity lookup to Isaac Lab's ``InteractiveScene``. It
does not copy those entities into another mapping or expose live counterparts of
Arena configurations.

``ArenaWorld`` exposes run-time facts; it does not implement task predicates.
The predicate library remains a collection of plain functions that combines
pose, geometry, contact, and motion queries into named Boolean relationships.
This follows RoboLab's useful separation between ``WorldState`` and
``predicate_logic`` without copying its global ownership.

Isaac Lab constructs ``env.scene`` before calling ``load_managers``. The Arena
environment creates ``arena_world`` at the beginning of its ``load_managers``
override, before calling ``super().load_managers()``, so it is available while
manager terms are constructed. The environment's ``close`` override releases
the facade's cached tensors and scene references before delegating to Isaac
Lab. Both creation and teardown are safe when initialization was only partial,
and repeated ``close`` calls are harmless.

Predicate interface
~~~~~~~~~~~~~~~~~~~

Manager-compatible predicates follow these rules:

* The first argument is the live manager-based environment.
* Whole scene entities are identified by ``str`` names.
* A contact predicate receives the existing contact sensor's scene name as a
  ``str`` when Arena cannot derive it from the object pair.
* Remaining arguments are plain scalar or collection configuration.
* One call evaluates every parallel environment.
* The result is a Boolean tensor with shape ``(num_envs,)``.
* Predicates do not accept Arena objects or return an optional scalar selected
  by ``env_id``.

Every whole-entity operand must be present under that name in the composed
Isaac Lab scene configuration and resolvable through ``env.scene[name]``.
Tasks that currently use an unregistered ``ObjectReference`` only to configure
a contact filter must register its runtime view before using it as a geometry
operand. Environment construction validates this requirement instead of
letting the first predicate call fail late.

``SceneEntityCfg`` remains appropriate for an Isaac Lab term that needs body,
joint, or sensor-index resolution. It is unnecessary when a predicate only
needs an entire entity by name.

Name resolution validates that the name exists and that the entity supports the
requested operation. Errors should identify the predicate, requested name, and
available compatible entities.

Live state
~~~~~~~~~~

Live poses, velocities, contacts, and joint state remain in ``env.scene``.
``ArenaWorld`` normalizes name-based access across the supported Isaac Lab
entity types, but it does not retain a second entity mapping or cache mutable
state.

For the initial rigid-object, rigid-object-set, and static-reference scope,
``get_pose_w`` returns one ``(num_envs, 7)`` world-frame tensor ordered
``(x, y, z, qx, qy, qz, qw)``. ``get_pose_env`` returns the same representation
after subtracting each environment origin from the position. Both follow this
checkout's Isaac Lab and Arena ``xyzw`` quaternion convention. The world pose
defines the geometry cache's pose frame; translation and rotation are applied
from live state on every predicate evaluation.

``get_linear_velocity_w`` returns the root linear velocity as a world-frame
``(num_envs, 3)`` tensor. The initial motion predicate does not read or cache
angular velocity.

Pose-frame geometry cache
~~~~~~~~~~~~~~~~~~~~~~~~~

``ArenaWorld`` privately stores immutable geometry derived from the composed
stage. A cached entry may contain:

* local AABB minimum and maximum points;
* local corners and centroid;
* convex-hull vertices, centroid, and planes; and
* the mapping from parallel environment to geometry variant.

Scale and USD unit conversion are baked into local geometry. Live translation
and rotation are not.

“Local” in the cache API means local to the explicitly resolved live pose frame,
not local to an Arena configuration, source USD file, or arbitrary ancestor.

The public lookup is by scene name. Internally, the cache stores geometry once
per unique resolved source and pose frame, and records which environment IDs
use it. ``get_local_hull_groups`` returns immutable groups containing one local
hull and the environment IDs that use it. A regular object normally has one
group; a heterogeneous object set can have several. Hulls are not padded into
one tensor because different sources can have different vertex and plane
counts.

Conceptually, each run-time-derived group contains:

.. code-block:: python

   LocalHullGroup(
       env_ids,          # (K,)
       vertices,         # (V, 3)
       centroid,         # (3,)
       full_planes,      # (F, 4), normal xyz followed by offset
       open_top_planes,  # (F_open, 4)
   )

This is a query result and cache value, not an authored configuration or a
run-time specification transferred from Arena objects.

When both predicate arguments have variants, the predicate intersects their
environment-ID groups, evaluates each non-empty intersection with the
corresponding pair of hulls, and scatters the results into one
``(num_envs,)`` Boolean tensor. Resolution validates that every environment is
covered exactly once for each entity, with neither gaps nor overlaps.

Isaac Lab's ``iter_clone_plan_matches`` already provides the required mapping
from an entity prim expression to source prims and environment IDs. It is also
used by Isaac Lab's multi-mesh ray caster, so Arena does not need to interpret
clone-plan rows itself. The resolver gets that prim expression from the
composed Isaac Lab scene configuration, not from an Arena configuration. A
direct-stage fallback handles a homogeneous scene without a clone plan.

Cache entries are populated lazily. They survive episode resets because object
topology and scale are assumed immutable for the live environment. ``ArenaWorld``
is closed and releases its cached device tensors before the environment releases
``env.scene``. Run-time topology or scale changes require explicit invalidation
and are not supported initially.

Geometry extraction
~~~~~~~~~~~~~~~~~~~

The first cache derives hulls from **composed, default-purpose mesh geometry**.
“Composed” means geometry in the live USD stage after OpenUSD has resolved
references, payloads, variants, overrides, and spawn scale. The extractor does
not read an Arena object's pre-simulation bounding box or assume that an
unopened source USD describes what was spawned.

For the first implementation, the executable selection rule is:

* traverse descendant ``UsdGeom.Mesh`` prims of each resolved composed source,
  including instance proxies;
* include a mesh when its computed USD purpose is ``default`` (which also
  covers an unset purpose); and
* ignore ``render``, ``proxy``, and ``guide`` representations.

USD purpose selects among model representations; collision participation is a
separate property. Therefore this rule does not mean “use collision geometry”
or “use visual geometry.” It follows RoboLab's default/unset mesh selection.
Arena deliberately improves on the reviewed ``f5acaa51`` traversal by including
instance proxies and using the computed purpose so inherited purpose metadata
is respected; current RoboLab also includes instance proxies.

If no selected mesh exists, or the collected points cannot form a 3D convex
hull, the first lookup fails with the entity name and the reason. There is no
silent fallback to a pre-simulation AABB. Analytic USD primitives and assets
authored only under another purpose are follow-up geometry sources; supporting
them requires an explicit conversion rather than an accidental mixture of
representations.

The entity's configured prim expression identifies the geometry traversal root;
it does not necessarily identify the rigid actor pose frame. For a rigid entity,
the resolver uses the concrete root-view prim that supplies ``root_pose_w`` as
the pose frame and transforms selected mesh points into that frame. For a static
reference, the matched reference prim itself is the pose frame. The cache key
includes both the resolved geometry source and pose-frame prim.

The extractor must not compute a world AABB and undo only translation: that
loses orientation information through repeated axis alignment. It instead
transforms local points or bounds fully into the selected pose frame before
computing the cached local result.

Rigid objects
^^^^^^^^^^^^^

A regular rigid object normally resolves to one composed source shared by all
environments. Its local geometry is cached once, while its root pose is read
from the live rigid-object view.

Object references
^^^^^^^^^^^^^^^^^

An object reference resolves through its own scene name and generated live prim
expression. Geometry is measured in the reference's live pose frame. The
run-time path does not need its ``parent_asset``, pre-simulation bounding box, or
initial transform relative to that parent.

Only static object references are in the first implementation. A reference view
can initially expose only source prims because it is created before scene
cloning. Resolution combines concrete matched prim paths, clone-plan ownership,
and environment origins to produce exactly one world pose per environment.
Moving references and references to articulated links are out of scope; their
pose cannot be reconstructed by replicating a static source transform.

Rigid object sets
^^^^^^^^^^^^^^^^^

For a ``RigidObjectSet``, clone-plan matches identify each unique spawned source
and the environment IDs populated from it. Geometry is extracted once per
source and selected per environment when the predicate evaluates.

No object-set member definitions, variant indices, or cached pre-simulation
bounds are transferred into the live environment.


Minimal viable implementation
------------------------------

The implementation starts from current ``main``. No PR #979 code is
cherry-picked, ported, or incrementally refactored. Its research and failure
modes inform this document only.

The first implementation is one vertical slice for spatial containment:

#. Replace repository uses of ``ObjectBase.get_object_pose`` and
   ``ObjectBase.set_object_pose``. Reads move to
   ``env.arena_world.get_pose_w(name)`` or ``get_pose_env(name)``; writes use the
   existing ``isaaclab_arena.terms.events.set_object_pose`` live-entity helper.
   Then remove both methods from the pre-simulation configuration class.
#. Add a public, environment-owned ``ArenaWorld`` to
   ``IsaacLabArenaManagerBasedRLEnv`` with fresh pose/contact queries and a
   private local-geometry cache.
#. Resolve composed geometry sources and their environment IDs with
   ``iter_clone_plan_matches``.
#. Register and validate runtime views for every predicate operand, including
   destination references currently used only to configure a contact filter.
#. Extract and cache local geometry in the live pose frame for regular rigid
   objects, static object references, and heterogeneous rigid object sets.
#. Implement the atomic predicates
   ``object_in_container(env, object_name, container_name)``,
   ``object_supported_by(env, object_name, support_name,
   contact_sensor_name, ...)``, and
   ``object_not_moving(env, object_name, ...)`` as plain functions returning
   one Boolean per environment.
#. Add one canonical
   ``object_stably_placed(env, object_name, destination_name,
   contact_sensor_name, ...)`` function that composes the three atomic
   predicates. Use the same callable and parameters for pick-and-place success
   termination and its corresponding final progress milestone.
#. Remove the unused ``PickAndPlaceTask.max_separation`` option. The geometric
   destination test supersedes its optional axis-aligned proximity check.
#. Do this without introducing ``ArenaAssetHandle`` or a manager-term adapter.
#. Add unit tests for hull construction and variant grouping, plus
   live-simulation tests for rotated and scaled geometry, a nested rigid-body
   pose frame, a static object reference, a heterogeneous object set, cache
   reuse, changing live poses, contact-force direction, representative
   container and surface destinations, and the composed placement condition.

This slice changes which predicate the existing pick-and-place progress
objective invokes, keeping that milestone consistent with task success. It does
not change ``ProgressTracker``, ``ProgressObjective``, the settling recorder, or
any progress-tracking lifecycle or public API.


Containment contract
--------------------

The first ``object_in_container`` implementation follows RoboLab's executed
geometry test: the arithmetic mean of the object's cached convex-hull vertices
is transformed into the container's live pose frame and tested against the
container's cached open-top convex-hull planes. Container faces whose outward
normal has a local ``+Z`` component of at least ``0.7`` are removed. This drops
cap-like faces within approximately 45 degrees of straight up and is intended
to model an open top. It does not mathematically guarantee that every upward ray
is unbounded, because retained sloped faces can still have a smaller positive
``Z`` component.

This is a centroid-in-convex-hull approximation, not full object containment. A
convex hull fills concavities, the object can partially protrude, and the opening
is assumed to face local ``+Z``. The predicate documentation and tests must state
those semantics explicitly.

The initial predicate has no geometry ``tolerance`` argument. RoboLab accepts
one for backward compatibility but does not apply it in the convex-hull test;
Arena should not expose a parameter that has no effect. A later tolerance must
have an explicit half-space or distance definition.

Authored cavity volumes, non-convex containment, and authored opening directions
belong to affordances and are follow-up work.


Pick-and-place destination integration
--------------------------------------

Arena's generic ``PickAndPlaceTask`` currently uses the same ``on`` transition
for placing an object on a surface and dropping it into an open container. The
first integration preserves that task model: the open-top hull acts as an
upward-open destination catchment for volumetric containers, plates, trays, and
shelves. For a thin support object, its lower and side planes provide the floor
and footprint after cap-like upward faces are removed; the independent support
predicate prevents a floating object from succeeding merely because its
centroid is in that catchment.

This deliberately differs from RoboLab, which exposes separate container and
surface placement conditionals. Arena's shared behavior is accepted only for a
non-degenerate 3D destination mesh and is verified on representative container,
plate, tray, and shelf assets. A coplanar surface fails hull construction rather
than silently using a different relationship. A later affordance or task-domain
change can split ``in`` and ``on`` semantics without changing ``ArenaWorld``.

``PickAndPlaceTask.max_separation`` has no call sites on current ``main`` and is
removed in this slice. Its axis-aligned center-distance check is neither the
containment contract nor the support-surface footprint contract, and retaining
it as an optional extra would let termination and progress drift again.


Stable-placement contract
-------------------------

Stable placement is a separate task-level conjunction:

.. code-block:: text

   object_stably_placed(object_name, destination_name, contact_sensor_name)
       = object_in_container(object_name, destination_name)
         AND object_supported_by(
                 object_name, destination_name, contact_sensor_name
             )
         AND object_not_moving(object_name)

This is a new Arena task-level composition. RoboLab implements analogues of the
three checks separately, but it does not combine all three as one placement
condition. Keeping Arena's atomic predicates explicit avoids a Boolean-flag API
that changes what one predicate means.

The task-level composition receives three scene-name strings: the object, the
destination, and the task-created filtered contact sensor. Pair-based contact
sensor lookup may be added independently if Arena later standardizes sensor
registration.

``object_supported_by`` validates that the named contact sensor represents one
sensor body belonging to ``object_name`` filtered against one body belonging to
``support_name``. ``ArenaWorld.get_filtered_contact_force`` then returns the
summed world-frame force **on the sensed object from that filtered support**,
with shape ``(num_envs, 3)``. A missing force matrix or any unsupported
many-body/many-filter shape fails explicitly.

Under the initial world-gravity convention of ``-Z``, the predicate requires a
configurable minimum magnitude, a positive world
``+Z`` component, and a force direction within a configurable cone around world
``+Z``. This follows RoboLab's ``is_supported_on_surface`` implementation and
tests physical support against gravity rather than destination-local
orientation. RoboLab can correct force sign from its deterministic pair name;
Arena's explicit sensor contract instead fixes the sensed-object side and tests
that direction directly.

``object_not_moving`` compares the object's current linear-speed norm with a
configurable threshold. It does not check angular velocity in the first
implementation, matching current Arena's ``object_on_destination`` and RoboLab's
optional stationary check for ``object_in_container``. Angular stability can be
added later under an explicit contract.


Verification
------------

Pure tests cover convex-hull construction, upward-plane removal, variant-group
intersection and scattering, quaternion transforms, and clear failures for
missing or degenerate geometry. Clone-resolution tests cover a homogeneous
scene without a clone plan and reject environment-mask gaps or overlaps.

Live-simulation tests prove the architecture rather than only the predicate
math. They cover a rotated and scaled rigid object, a registered static
``ObjectReference``, a heterogeneous ``RigidObjectSet``, and the filtered-force
shape and direction contract. They also verify that:

* an object touching the container's exterior is not contained;
* strong lateral contact is not upward support;
* a moving object is not stably placed;
* a pose change affects the next evaluation without rebuilding geometry;
* an episode reset reuses static geometry;
* sequential ``ArenaWorld`` instances do not share cache entries; and
* close, repeated close, failed manager construction, and reconstruction release
  and rebuild the cache safely.


Rejected alternatives
---------------------

Pass Arena configurations to predicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This crosses the pre-simulation boundary, depends on copied or preserved Python
identity, and combines configuration-time geometry with live state.

Expose Arena configurations from the live environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An environment-owned definition registry moves the same coupling to a central
location. Predicates still depend on pre-simulation objects and their caches,
and object-set geometry must still be reconciled with the composed scene.

Introduce ``ArenaObjectRuntimeSpec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometry, source assignment, scale, and reference frames can be derived from
the composed stage and clone plan. A transferred specification duplicates that
information and introduces another object model without solving a current need.

Use a module-global ``WorldState``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name-based API and geometry caching are useful, but a module-global facade
does not give caches an unambiguous environment lifetime and cannot safely serve
multiple live environments.

Attach Arena state to ``InteractiveScene`` entities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``InteractiveScene`` is an Isaac Lab runtime abstraction and already owns live
entities. Adding Arena-specific fields couples upstream types to Arena and still
does not define per-variant geometry or cache lifecycle.


Constraints
-----------

* Arena objects, references, sets, and scenes remain pre-simulation
  configuration.
* Two live Arena environments never share or replace each other's cache.
* Poses and other mutable state are always read from the live scene.
* Geometry is not traversed from USD during every simulation step.
* Every environment uses the geometry variant actually spawned there.
* Predicate configuration contains names and ordinary values, not Arena object
  identity.
* The design does not change placement, variant assignment, or cloning.


Non-goals for the first implementation
--------------------------------------

* Changes to progress-tracking architecture, lifecycle, or public API.
* Mimic environment support.
* A general framework for episode-lifetime predicate state.
* Normalizing every existing predicate signature.
* Articulated-object containment. A moving articulation cannot use one static
  root-local box for all of its links.
* Deformable-object containment. Its current geometry must come from live nodal
  state rather than an environment-lifetime static cache.
* Run-time geometry topology or scale changes.
* Analytic or non-default-purpose USD geometry.
* Coplanar destination geometry without a 3D convex hull.
* Authored affordance volumes or opening directions.
* Requiring gripper detachment for stable placement. This first requires a
  consistent gripper-contact interface across embodiments.


Follow-up work
--------------

After the first spatial slice is validated, independent changes can:

* adopt the predicate in additional progress objectives;
* move settling history to explicit episode-lifetime predicate state;
* normalize existing predicates to the string-based, batched contract;
* add full-object, non-convex, and affordance-defined containment;
* add further spatial predicates such as footprint, directional, and proximity
  relationships using the same live-state and cached-geometry boundary;
* add an ``object_detached_from_gripper`` predicate and include it in task-level
  placement once embodiments expose a consistent gripper-contact interface; and
* add articulated and deformable geometry paths where their live state requires
  different cache semantics.
