Runtime Arena Asset Lookup
==========================

**Status:** Discussion

**Scope:** How a run-time predicate finds both Arena geometry metadata and live
Isaac Lab state for a named scene element. This document does not choose between
bounding-box and mesh containment, or change placement and cloning behavior.

Why this is needed
------------------

A spatial predicate such as ``object_in_container`` needs two kinds of data:

* static geometry, such as the object's cached local bounding box; and
* live state, such as the object's pose in every parallel environment.

Arena currently keeps those in different models. An Arena asset definition owns
construction settings and cached geometry. The corresponding Isaac Lab scene
entity in ``env.scene`` owns the live, batched simulation state. Looking up only
the Isaac Lab entity is sufficient for position or velocity predicates, but not
for containment because its Arena geometry and object-reference semantics are no
longer directly available.

``RigidObjectSet`` adds another requirement. One scene name can represent a
different spawned variant in each environment, so the predicate must combine the
live pose with the bounds of the member that was actually spawned there.

The goal is a plain, name-based predicate API without losing that information or
introducing process-global state.

The two object models
---------------------

The distinction between the two models should remain explicit:

``Arena asset definition``
   The object used while composing an Arena scene. It knows the USD source,
   scale, object-reference relationship, object-set members, and lazily cached
   local geometry.

``Isaac Lab scene entity``
   The object created when the environment starts. It exposes poses, velocities,
   contacts, and other live state for all parallel environments.

Both normally use the same scene name, but neither can replace the other. A
run-time lookup needs to associate them.

How RoboLab handles this
------------------------

RoboLab presents a simple string API:

.. code-block:: python

   object_in_container(env, "object", "container")

The simplicity is provided by a ``WorldState`` facade behind the predicate. The
relevant flow is:

.. code-block:: text

   predicate receives names
       -> get the WorldState for env
       -> resolve each name to an Isaac Lab scene entity
       -> load and cache local geometry under that name
       -> read the entity's live pose
       -> transform the object centroid into the container frame
       -> test the centroid against the container hull planes

The local bounding box and convex hull are read lazily from environment zero's
USD prim and cached by name. Live poses are read from Isaac Lab assets or XForm
views on every evaluation. Open-top containment removes the upward-facing hull
plane, so an object above the opening still satisfies the geometric test.
``object_on_top`` is a separate predicate: it combines an upward contact-force
cone with a centroid check in the support surface's XY footprint.

This is not a registry of RoboLab domain objects. It is a name-based service that
combines Isaac Lab entities, USD geometry queries, and caches. The implementation
reviewed here is in RoboLab commit ``f5acaa51``, principally in
``robolab/core/world/world_state.py`` and
``robolab/core/task/predicate_logic.py``.

The approach gives callers a clean API, but its current implementation has
constraints that do not fit Arena directly:

* ``WorldState`` is a module-global singleton that is replaced when the active
  environment changes. This makes independent live environments difficult to
  reason about.
* Geometry is cached by scene name after reading environment zero. This assumes
  that one name has the same geometry in every environment.
* RoboLab has no equivalent of Arena's ``RigidObjectSet`` in this path, so it
  does not solve per-environment spawned variants.
* Cache ownership and invalidation are implicit in the global facade.

The useful part to copy is therefore the name-only consumer API, not the global
singleton.

How PR #979 works now
---------------------

The current implementation keeps the predicate itself as a plain function but
passes Arena asset definitions to it:

.. code-block:: text

   manager TermCfg
       -> ArenaAssetHandle
       -> manager-term adapter
       -> plain spatial predicate
          -> cached bounds from the Arena asset definition
          -> live pose from env.scene[asset.name]
          -> spawned object-set member from the live clone plan

``ObjectBase.get_object_pose(env)`` is the bridge for live state. It resolves the
asset's scene name in ``env.scene`` and reads the pose from the live Isaac Lab
entity. Bounding boxes remain lazily cached on ``Object`` and
``ObjectReference``. For ``RigidObjectSet``, the current branch derives the
member present in each environment from the live scene's spawn configuration and
clone plan before selecting that member's cached bounds.

The ``ArenaAssetHandle`` exists only at the manager-configuration boundary.
Isaac Lab configuration classes and managers deep-copy their configuration, so
placing an Arena asset directly in term parameters produces a separate copy. The
handle's ``__deepcopy__`` keeps the term connected to the original asset
definition and its caches; the adapter unwraps it before calling the predicate.

This approach is deliberately local and works without changing environment
construction. Its drawbacks are visible in the task definitions:

* term parameters contain live Python object identity instead of plain data;
* every task that uses the predicate must construct handles and use an adapter;
* a reader must understand why one particular parameter defeats deep-copying;
* the association between an Arena definition and an Isaac Lab entity is rebuilt
  indirectly inside each consumer.

Passing the Arena asset directly would remove the small wrapper but not the
underlying issue. The manager would still copy a non-plain domain object, which
would duplicate its caches and make it unclear whether later definition changes
are visible. It is a less explicit version of the same coupling.

Possible designs
----------------

Keep the current per-term handles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the smallest change and keeps PR #979 self-contained. It is reasonable
if containment is the only consumer and the architecture is intentionally
deferred. It should not become the default pattern for every predicate that
needs Arena metadata.

Copy RoboLab's global ``WorldState``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predicates would accept names and ask a module-global facade for geometry and
live state. The call site would be simple, but multiple-environment ownership,
cache invalidation, and ``RigidObjectSet`` handling would be hidden in global
state. This is not recommended.

Expose Arena asset definitions on each live environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The environment can own a read-only name-to-definition lookup:

.. code-block:: python

   def object_in_container(
       env,
       object_cfg: SceneEntityCfg,
       container_cfg: SceneEntityCfg,
   ) -> torch.Tensor:
       ...

   object_definition = env.unwrapped.arena_asset_definitions[object_cfg.name]
   container_definition = env.unwrapped.arena_asset_definitions[container_cfg.name]

The deliberate name ``arena_asset_definitions`` makes clear that the returned
objects are not live Isaac Lab entities. The predicate would still use the live
environment when asking a definition for its pose or spawned bounds:

.. code-block:: python

   object_pose_w = object_definition.get_bounding_box_pose(env, is_relative=False)
   object_bounds = get_spawned_bounding_box_per_env(object_definition, env)

Manager term parameters can then contain only ``SceneEntityCfg`` values. The
Isaac Lab manager validates those scene names, while the predicate uses the same
names to retrieve Arena metadata. Static geometry remains cached once on the
original definitions, and live poses still come from ``env.scene`` on every
evaluation.

This lookup does not turn an Arena definition into a live object. It only makes
the matching definition reachable by name; methods that need state still receive
the environment explicitly. A bound run-time view, described below, would be the
larger abstraction that exposes pose and bounds directly.

This is the smallest design that provides RoboLab's name-only predicate API
without a global singleton. It moves the identity bridge from every term to one
environment-construction boundary.

There is one important construction detail. Gymnasium deep-copies registered
environment keyword arguments before calling the environment constructor. The
usual ``ArenaEnvBuilder.make_registered`` path also supplies the original
``env_kwargs`` directly to ``gym.make``; those explicit values replace the
registered copies and retain their identity. Arena additionally supports a
registration-only flow in which another system later calls ``gym.make`` without
those explicit values. That path receives a copy of the registered mapping.

A copied mapping is not necessarily incorrect. If it is a finalized snapshot
created after relation solving and build-time variations, it can own independent
lazy geometry caches. Preserving original identity is required only if later
definition mutations must remain visible or copying and rebuilding geometry
caches is unacceptable.

Three implementation choices are possible at that boundary:

* Accept Gymnasium's copy as a finalized run-time snapshot. This requires no
  identity-preserving wrapper, but duplicates the definition graph and its lazy
  caches in the registration-only path.
* Pass one read-only, identity-preserving collection through Gymnasium. This is
  still a special deep-copy rule, but it exists once, at the exact registration
  boundary, rather than once for every asset in every manager term.
* Convert each definition to a plain, deep-copy-safe run-time specification and
  pass those specifications instead. This removes identity requirements but is a
  larger change because all required geometry and object-reference metadata must
  be represented explicitly.

The environment-owned lookup must be installed for both the normal Arena
environment and embodiment-specific Mimic environments. Adding a field only to
``IsaacLabArenaManagerBasedRLEnv`` would miss the Mimic entry points, which
currently inherit directly from Isaac Lab's ``ManagerBasedRLMimicEnv``. A shared
Arena mixin or construction hook is required, and the lookup must be available
before manager terms can run.

Expose bound run-time asset views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fuller facade could return an environment-bound view instead of the original
definition:

.. code-block:: python

   object_view = env.unwrapped.arena_scene[object_cfg.name]
   object_pose_w = object_view.pose_w
   object_bounds = object_view.local_bounds

Each view would bind an Arena definition, its Isaac Lab scene entity, and the
spawned variant for each environment. This creates the cleanest consumer API and
provides one place to normalize ``ObjectReference`` poses and cache object-set
variant selection. It also introduces a new run-time abstraction with lifecycle,
typing, and invalidation rules. It is worthwhile if many systems need combined
geometry and state, but it is more than the containment predicate currently
requires.

Attach Arena metadata to Isaac Lab entities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arena could copy its bounds and reference metadata onto the generated Isaac Lab
configuration or entity. Consumers would then need only ``env.scene[name]``.
This tightly couples Arena semantics to Isaac Lab classes and still needs a
representation for per-environment object-set variants. It is not preferred.

Make the predicate a stateful manager term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``ManagerTermBase`` subclass could resolve and cache dependencies once. That
may reduce repeated lookup, but it does not provide the missing name-to-Arena
association. Every such term would otherwise recreate its own small registry, so
this is an optimization choice rather than the architectural solution.

Recommended direction
---------------------

Use an environment-owned lookup of Arena asset definitions as the next,
separate refactor. Keep predicates as plain functions and keep their manager
parameters as ``SceneEntityCfg`` values.

The proposed flow is:

.. code-block:: text

   Arena Scene.assets
       -> finalize the scene definition
       -> pass one read-only definition collection through environment construction
       -> create the Isaac Lab scene and clone plan
       -> bind and validate names for this live environment
       -> let predicates resolve definitions by SceneEntityCfg.name

The first version should remain small:

#. Add a read-only ``arena_asset_definitions`` lookup owned by each live Arena
   environment.
#. Share its construction path between the normal and Mimic environment classes.
#. Treat the mapping as a finalized snapshot. Initially either allow the
   registration-only path to copy it, or preserve the collection only at that
   Gymnasium boundary if duplicated geometry caches are measurably expensive.
   Introduce plain run-time specifications only if serialization is already a
   requirement.
#. Change spatial manager terms to pass only ``SceneEntityCfg`` values and remove
   the per-term ``ArenaAssetHandle`` adapters.
#. Continue reading poses from the live Isaac Lab scene on every predicate call.
#. Continue selecting ``RigidObjectSet`` geometry from the live clone plan; move
   that selection into a bound run-time view only if more consumers need it.

This provides the simplicity of RoboLab's string lookup while keeping state
scoped to the environment and retaining Arena's richer asset model.

Design constraints
------------------

Any chosen design should satisfy these constraints:

* Two simultaneously live environments must not replace or invalidate each
  other's lookup or caches.
* A pose must always be read from live simulation state, never from the asset's
  initial pose.
* Static geometry must not be re-read from USD on every simulation step.
* A ``RigidObjectSet`` must use the member actually spawned in each environment.
* Normal and Mimic environments must expose the same lookup behavior.
* Predicate configuration should contain names and scalar settings, not Arena
  object identity.
* Introducing the lookup must not alter placement, variant assignment, or clone
  behavior.

Open questions
--------------

* Is the environment-owned definition lookup intended only for in-process Arena
  construction, or must an environment be reconstructible from serialized Gym
  configuration alone? This decides between one construction-boundary wrapper
  and plain run-time specifications.
* Should the public API stop at ``arena_asset_definitions[name]``, or are enough
  future consumers expected to justify a bound ``arena_scene[name]`` view now?
* Are spawned variants immutable for the lifetime of an environment? If not, a
  bound view needs an explicit invalidation point when variants change.
* Should the registry initially include every Arena asset, or only
  ``ObjectBase`` instances that have a corresponding Isaac Lab scene entity?
