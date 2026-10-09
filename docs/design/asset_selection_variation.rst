:orphan:

Asset selection as a variation
==============================

Status: design draft, 9 October 2026. The draft implementation supports
the Python API below. Build manifests and replay remain future
work. These changes are not available on ``main`` yet.

The Object refactor is tracked in `#1424
<https://github.com/isaac-sim/IsaacLab-Arena/pull/1424>`_, on top of #1419.
Asset selection builds on that refactor and implements the Python API and
fixed per-environment episode recording. Build manifests, validated selection
replay, YAML candidate authoring, and caller migration are still future work.
Selection currently rejects variation and placement replay explicitly.

Use one generic ``Object`` for both a fixed asset and asset selection. The
object keeps its role in the scene, such as ``pick_up_object``. It may receive
an asset definition at construction, or an enabled ``AssetSelectionVariation``
may assign its asset during the build.

Asset assignments last until rebuild. The replay proposal below assumes
matching environment and candidate definitions, the original environment
count, and the recorded assignments.
Candidates are native definitions represented by ``tuple[str, SpawnerCfg]``.
Their names identify them in recordings, and sequential assignment is the
default. Random assignment remains an optional sampler. A small registry
adapter obtains definitions from existing rigid library constructors; it
does not introduce another class or redesign the library.
The shared ``sample_per_environment`` setting is implemented alongside the
Python API. The replay format below remains a proposal for review.

Motivation
----------

Imagine a task that asks a robot to pick up a fruit and put it in a bowl.
We want to evaluate the same task with bananas, oranges, and lemons. Running
those objects in parallel gives us variety within one simulation build.
The task should still refer to ``pick_up_object`` regardless of which fruit
appears in a particular environment.

``RigidObjectSet`` already provides that behavior. Its assignments are fixed
across resets: an environment built with an orange keeps its orange until
rebuild. Moving selection into variations does not change that lifetime.
It changes how users describe, enable, record, and replay the choice.

Today, ``RigidObjectSet`` cycles through its members in order by default.
Its optional ``random_choice=True`` samples independently for each
environment. We retain both behaviors, with sequential assignment still
the default and the sampler configuration selecting between them.

Today, asset selection has its own authoring and assignment path, separate
from variations. A user can enable lighting or mass variation through the
experiment configuration, but choosing the fruit requires a different scene
construct. Recording the other sampled values also does not, by itself,
explain which fruit an episode used.

Treating asset selection as a variation brings these decisions together.
An environment author declares suitable candidates once. An experiment can
then use the default fruit, select one fruit for the whole build, or select
a fruit separately for each environment. The task and its object reference
stay the same. Episode results identify the selected candidate alongside
the other variation values.

This also gives us a clearer way to retire the separate set API. We still
need the machinery that resolves assignments, prepares compatible assets,
and supplies geometry to placement. That work can live behind ``Object``
and the builder, without making users choose a second object class.

High-level goals
----------------

* **Keep one scene identity.** Tasks, relations, reset behavior, and metrics
  refer to the same named object before and after asset selection.
* **Make selection an ordinary variation.** It appears in variation
  discovery, experiment overrides, and recording.
* **Support diversity within one build.** Different environments can use
  different candidates while retaining the same task structure.
* **Keep placement and simulation consistent.** Both use the same selected
  native asset configurations, including scale and physical settings.
* **Make assignment explicit.** A default asset remains usable when selection
  is disabled. Without a default, an enabled variation must assign an asset
  before placement; the first candidate is never an implicit fallback.
* **Make recorded choices explicit.** Episode records identify the selected
  asset. Restoring those choices is covered by the future replay proposal.

The first version supports rigid candidates with one rigid body each.
It does not switch assets on reset, mix rigid objects with articulations or
deformables, or change the robot or task topology. It also does not promise
that every existing property variation automatically works with asset
selection. Those combinations need their own clear behavior. The only
assignment samplers planned here are sequential and random; balanced random
sampling is outside this first version.

What an object represents
-------------------------

``pick_up_object`` is a generic ``Object`` used by the scene and task.
``Object(name="pick_up_object", asset=banana_asset)`` supplies its normal default.
``Object(name="pick_up_object")`` leaves asset assignment to a variation.
Both use the same class and geometry interface.

``banana_asset`` and ``orange_asset`` are reusable native asset definitions.
Each is a ``tuple[str, SpawnerCfg]`` containing an ID and native spawn settings.
``AssetRegistry.get_asset_definition(name, **constructor_overrides)`` obtains
one by calling the existing rigid library constructor and copying its native
spawn settings. The tuple retains the requested registry ID regardless of
the constructed instance's name. The variation copies these definitions
when declared.
Candidate definitions are not scene nodes. The task keeps its reference to
``pick_up_object``.

For example, a build with three parallel environments could resolve to:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Environment
     - Scene object
     - Assigned candidate
   * - 0
     - ``pick_up_object``
     - Banana
   * - 1
     - ``pick_up_object``
     - Orange
   * - 2
     - ``pick_up_object``
     - Banana

When environment 1 resets, placement may choose a new pose for its orange.
The asset assignment does not change. A later rebuild may choose a different
assignment.

The scene object owns its name, scene path, pose, relations, reset behavior,
and variations. Each definition's name becomes its recording ID; it does
not rename the scene object. The registry adapter transfers no subclass
behavior, poses, relations, or variations. Settings on the enclosing Isaac
Lab asset configuration remain the scene object's responsibility; callers
can supply ``asset_cfg_addon`` to ``Object`` as before. ``Object(name)`` is a
rigid declaration in this first scope. Its mass and disappearance variations
remain discoverable before build overrides, even while its asset is unassigned.

Keep existing library constructors and their behavior available for existing
callers. This change adds only an adapter to obtain native definitions and
the two generic ``Object`` construction forms. It does not rewrite library
classes, move affordances into a new model, or transfer fruit-specific methods
onto a selected object. Generic rigid-object tasks are the first supported
use case; candidate-specific behavior remains separate future work.

User API
--------

Declare the object and its choices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``background``, ``table``, and ``bowl`` have already been created by the
environment definition. Obtain native definitions, then declare the scene
object without a default so the enabled variation supplies its asset:

.. code-block:: python

   from isaaclab_arena.assets.object import Object
   from isaaclab_arena.assets.registries import AssetRegistry
   from isaaclab_arena.relations.relations import On
   from isaaclab_arena.scene.scene import Scene
   from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
   from isaaclab_arena.variations.asset_selection_variation import (
       AssetSelectionVariation,
       AssetSelectionVariationCfg,
   )
   asset_registry = AssetRegistry()
   banana_asset = asset_registry.get_asset_definition("banana_ycb_robolab")
   orange_asset = asset_registry.get_asset_definition("orange_01_fruits_veggies_robolab")

   pick_up_object = Object(name="pick_up_object")
   pick_up_object.add_relation(On(table))
   pick_up_object.add_variation(
       AssetSelectionVariation(
           asset_candidates=[banana_asset, orange_asset],
           cfg=AssetSelectionVariationCfg(enabled=True),
       )
   )

   scene = Scene(assets=[background, table, bowl, pick_up_object])
   task = PickAndPlaceTask(
       pick_up_object=pick_up_object,
       destination_location=bowl,
       background_scene=background,
   )

``add_variation()`` binds the variation to its target, so callers do not
repeat ``object=pick_up_object``. Each scene object can declare at most one
asset-selection variation.

The example enables selection immediately. With four environments, it assigns
banana, orange, banana, orange.

To give the same object a normal default, construct it with an asset definition:

.. code-block:: python

   pick_up_object = Object(name="pick_up_object", asset=banana_asset)

The config argument to ``AssetSelectionVariation`` is optional. Omitting it
declares the variation as disabled, with one value per environment and
sequential assignment. An object with a default then keeps that asset until
selection is enabled. Without a default, a missing or disabled assignment
variation is an error. The builder checks every object's assignment after
overrides and variations, before placement. It never takes the first candidate
as an implicit default. A Python caller can enable a disabled variation later:

.. code-block:: python

   pick_up_object.get_variation("asset_selection").enable()

Candidate names supply the identifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no second set of dictionary keys to declare. Each definition's name
is copied as its candidate ID, alongside its native spawn settings.
In the example, those names are ``banana_ycb_robolab`` and
``orange_01_fruits_veggies_robolab``. Names must be nonempty and unique within
the list; duplicate names are an error, even if the configurations match.

Two differently scaled oranges need distinct candidate names:

.. code-block:: python

   _, small_orange_spawn_cfg = asset_registry.get_asset_definition(
       "orange_01_fruits_veggies_robolab",
       scale=(0.8, 0.8, 0.8),
   )
   _, large_orange_spawn_cfg = asset_registry.get_asset_definition(
       "orange_01_fruits_veggies_robolab",
       scale=(1.2, 1.2, 1.2),
   )
   small_orange_asset = ("small_orange", small_orange_spawn_cfg)
   large_orange_asset = ("large_orange", large_orange_spawn_cfg)

The scale overrides go to the existing constructor. Both registry calls
return ``orange_01_fruits_veggies_robolab`` as their ID; the explicit tuples
give the two candidates distinct IDs. Neither renaming a Python variable
nor passing ``instance_name`` to the constructor changes the returned registry
ID. Later edits to a supplied spawn configuration do not change the variation's
copied candidate definition.

The list must be nonempty, and every element must be a named native definition
for one concrete rigid asset. Its order controls sequential assignment;
recorded values use the declared tuple IDs. A target's default, when provided,
does not have to appear among its candidates.

Use the common variation configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``AssetSelectionVariationCfg`` extends ``VariationBaseCfg``. Native candidate
definitions stay on the variation constructor; the config contains the
experiment settings that users can inspect and override.

The common fields are:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Field
     - Selection default
     - Meaning
   * - ``enabled``
     - ``False``
     - Whether to apply the variation.
   * - ``sample_per_environment``
     - ``True``
     - Produce one value for each environment, or one value shared by all.
   * - ``sampler_cfg``
     - ``SequentialChoiceSamplerCfg()``
     - How to choose values from the ordered candidate list.

``sample_per_environment`` joins ``enabled`` and ``sampler_cfg`` in the base
configuration as a common setting. A sampler can produce a
deterministic sequence; sampling does not necessarily mean randomness.

Use ``SequentialChoiceSamplerCfg`` for A, B, C, A, B, C.
Use the existing ``ChoiceSamplerCfg`` for independent random choices:

.. code-block:: python

   from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg

   selection_cfg = AssetSelectionVariationCfg(
       enabled=True,
       sampler_cfg=ChoiceSamplerCfg(),
   )

The sampler is the only place that selects random versus sequential
behavior. There is no additional ``random_choice`` flag or distribution
setting on the variation.

Timing and value sharing remain separate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The variation's build-time or runtime lifecycle says when to choose values.
``sample_per_environment`` says whether environments get individual values.
``sampler_cfg`` says how those values are chosen. Asset selection is always
build-time in this first version, regardless of either config setting.

Existing variations keep their current behavior as their default. Shared
build-time lighting variations use ``sample_per_environment=False``;
per-environment runtime mass variation uses ``True``. Each variation declares
which settings it supports and rejects unsupported combinations. A common
field does not make a shared HDR light independently configurable or define
new shared sampling behavior for asynchronous runtime resets.

Configure an experiment
^^^^^^^^^^^^^^^^^^^^^^^

The environment defines the candidate assets. An experiment chooses how to
use them through the existing dotted variation overrides. The examples assume
the target's actual scene name is ``pick_up_object``; a Python variable with
that name does not by itself change the override path. This fragment belongs
under a run in an experiment configuration:

.. code-block:: yaml

   environment_builder:
     num_envs: 64
   variations:
     pick_up_object.asset_selection.enabled: true

With the default sequential sampler, the 64 environments alternate between
the banana and orange in list order. Choices stay fixed across resets. The
sequence starts from the first candidate at every build; it does not carry
a hidden counter between builds.

Random selection remains available by supplying ``ChoiceSamplerCfg`` as
shown above. A new build can then choose different assignments. Random
sampling does not guarantee equal counts or that every candidate appears.

Setting ``sample_per_environment=False`` requests one value and shares it
across the build. The sequential sampler then chooses the first candidate;
the random sampler chooses one candidate at random. Setting ``enabled=False``
makes no selection draw and leaves a configured default in use. Without a
default or another asset assignment, the build fails before placement.

To use one specific asset everywhere while recording that selection, declare
a single candidate:

.. code-block:: python

   selection = AssetSelectionVariation(
       asset_candidates=[orange_asset],
       cfg=AssetSelectionVariationCfg(enabled=True),
   )

This is an alternative declaration for the target, not a second selection
variation to attach alongside the earlier one. No separate fixed-selection
class or candidate-subset configuration is needed for the first version.
Changing the candidate list creates a different definition. Selection replay
is not supported in this implementation.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Configuration
     - What is selected
     - What happens on reset
   * - Disabled, with a default
     - Default asset in every environment
     - Keep the default asset
   * - Disabled or absent, without a default
     - Build fails before placement
     - No simulation is created
   * - Enabled, ``sample_per_environment=False``
     - One candidate for the whole build
     - Keep that candidate
   * - Enabled, ``sample_per_environment=True``
     - One candidate per environment
     - Keep each environment's candidate

Future YAML authoring
^^^^^^^^^^^^^^^^^^^^^

YAML candidate authoring is not part of this change. A later graph schema
can keep scene identity under ``objects`` and asset choices under the object's
variations. It should resolve a registry name and constructor overrides into
the same ``tuple[str, SpawnerCfg]`` definitions accepted by the Python API.
Candidate entries would not create scene nodes or another asset-library model.

That future schema must distinguish an optional default from the ordered
``asset_candidates`` list. Omitting the default must retain the same build-time
assignment requirement as ``Object(name)``. Relations and task arguments still
refer to the scene object. No separate object-set category or automatic
first-candidate default is needed. The exact YAML schema remains undecided.

Recording and future replay
---------------------------

Episode records already include fixed candidate IDs. The build manifest,
``build_id``, definition fingerprints, and replay behavior below are proposed
future work. Enabled selection currently rejects variation and placement replay.

Record the build and the episodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An episode row should make the choice easy to inspect. It only needs that
episode's candidate ID, rather than the assignment vector for the whole
simulation. Here is an abbreviated proposed episode row:

.. code-block:: json

   {
     "build_id": "fruit-evaluation-rebuild-0",
     "env_id": 1,
     "episode_in_env": 3,
     "variations": {
       "pick_up_object.asset_selection": "small_orange"
     }
   }

However, episode rows alone cannot describe the complete build. A slot may
never finish an episode, an episode budget may leave some slots unused, or
someone may filter the results to retain only failures. We must still know
which assets were built in all of those slots.

Write one companion build manifest before rollouts begin. For an episode
file named ``episode_results_rebuild0.jsonl``, the proposed companion is
``episode_results_rebuild0.build.json``. It contains the full assignment and
the fingerprints needed to validate supplied definitions. For example:

.. code-block:: json

   {
     "schema_version": 1,
     "build_id": "fruit-evaluation-rebuild-0",
     "num_envs": 3,
     "environment_definition": {
       "factory": "my_benchmark.fruit.FruitPickAndPlaceFactory",
       "config_type": "my_benchmark.fruit.FruitPickAndPlaceCfg",
       "config_fingerprint": "sha256:<environment-config-digest>"
     },
     "object_definitions": {
       "pick_up_object": {
         "default_asset_fingerprint": "sha256:<banana-config-digest>",
         "candidate_fingerprints": {
           "banana_ycb_robolab": "sha256:<banana-config-digest>",
           "small_orange": "sha256:<small-orange-config-digest>",
           "large_orange": "sha256:<large-orange-config-digest>"
         }
       }
     },
     "build_variations": {
       "pick_up_object.asset_selection": {
         "sample_per_environment": true,
         "samples": ["banana_ycb_robolab", "small_orange", "large_orange"]
       }
     }
   }

The digest strings are placeholders. With ``sample_per_environment: true``,
an entry has exactly ``num_envs`` samples, indexed by environment ID. With
``sample_per_environment: false``, it has one shared sample. Episode values
must agree with the manifest for their source environment.

Variation keys use the target's scene name. Sample values use the candidate
names copied when the variation was declared. ``candidate_fingerprints``
remains a mapping in the recording, but Arena generates its keys from those
names; authors do not supply a second set of candidate IDs.

``object_definitions`` includes every object that declares asset selection,
even when that variation is disabled. ``default_asset_fingerprint`` describes
the configured default before selection is applied, or is null when there is
no default. It is independent of the candidate list. ``build_variations``
includes all enabled build-time variations, including existing shared
variations. Their values can still appear in episode rows for convenient
analysis; both representations must agree. Disabled selection has no sample entry.

Use a separate JSON file because the current JSONL reader treats each object
as an episode and the runner uses those rows to determine the replay budget.
Adding a metadata row to that file would change both behaviors.

Keep the existing ``recorded_variation_samples_path`` input. By default,
replay discovers the companion next to that file. A proposed optional
``recorded_build_manifest_path`` supports renamed or filtered episode files:

.. code-block:: yaml

   environment_builder:
     num_envs: 3
     recorded_variation_samples_path: recordings/failures.jsonl
     recorded_build_manifest_path: recordings/episode_results_rebuild0.build.json
   variations:
     pick_up_object.asset_selection.enabled: true

Filtering episodes must retain their ``build_id`` and source ``env_id`` and
keep the original manifest. Missing assignments are errors; replay must not
guess them or sample replacements. The first version replays one recorded
build at a time. Every row must match the manifest's ``build_id`` and have an
integer environment ID between zero and ``num_envs - 1``.

Require matching definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recording validates an environment supplied by the caller. It does not
contain enough information to reconstruct the scene or candidate assets by
itself.

Record the registered factory identity, configuration type, and a fingerprint
of its effective declared configuration, including explicit defaults. For
graph definitions, normalize and fingerprint the graph content as part of
that configuration rather than hashing only its filename. For Python
definitions, require a serializable configuration; construction inputs
hidden outside that configuration are not covered by the check. Definitions
that cannot expose those inputs do not support validated replay in this
first version.

Use a canonical encoding of declared data, with sorted mapping keys and
explicit type identities. Do not hash a live Python object's ``repr()`` or
arbitrary internal state. An author can include a definition revision in the
configuration when implementation changes need to invalidate old recordings.
Retain repository revisions as provenance, but do not reject replay solely
because an unrelated policy or documentation commit changed the repository.

Policy choice, output paths, episode budget, and sampling seeds are not part
of environment-definition identity. Validate environment count separately.
Sampler settings belong to the replay eligibility rules below, rather than
the environment-definition fingerprint. This keeps policy comparisons and
restoring choices with a different seed possible.

Independently fingerprint each candidate's effective authored native spawn
configuration. Include its source identity, USD variant selections, scale,
materials, and spawn physics settings. Compute this before introducing
generated USD cache paths, local environment namespaces, or clone paths.
Use the same versioned encoding rules, with native types and supported
importable functions identified by qualified name. An unsupported opaque
field must be reported instead of omitted. This does not serialize the
implementation of a Python function.

Validate the full candidate name-to-fingerprint mapping and the target's
optional default as part of the object definition. Candidate list
order belongs to sampling policy, so normalize these definitions by captured
name for replay validation, including when fingerprinting a graph. Otherwise
an ordinary list hash would reject a harmless reorder.

Fresh sequential assignment follows the declared list order; replay resolves
recorded names to the current internal indices. Removing or renaming a
candidate, changing an asset setting, or changing ``sample_per_environment``
fails before placement. Duplicate candidate names are rejected at declaration.

Changing the seed or sampler is allowed because replay does not draw new
choices. Every restored name must exist in the validated candidate list,
and a recorded selection must remain enabled. Disabling it cannot silently
substitute a configured default or leave the target unassigned.

A configuration fingerprint cannot detect that a remote USD was replaced
at the same URL. Asset contents are only covered when the source carries an
immutable revision or a content digest. Matching factory identity and
configuration also cannot prove that Python behavior is unchanged. An
author-declared definition revision helps only when authors maintain it.
Replay restores the recorded asset choices and variation values; it does
not guarantee identical trajectories or unrecorded initialization.

Restore assignments before replaying episodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First validate the definitions and environment count. Then restore all
assignments from the manifest, derive geometry, and solve or restore
placement. Only after that should Arena construct the simulation.

Episode dispatch also needs to respect those assignments. The current
replay scheduler hands the next record to whichever environment requests
one. That would eventually hand an orange episode to a banana environment.
For the first heterogeneous version, keep each record on its source
environment and give every environment its own queue.

Preserve the runner's existing budget behavior explicitly:

1. Without an explicit episode budget, select every retained row once.
2. With a budget of ``K``, select the first ``K`` rows from the file sequence,
   cycling through that sequence when ``K`` exceeds its length.
3. Partition those selected occurrences by their source environment ID.
4. Let each environment drain its queue. Environments with no remaining
   records become inactive and do not contribute episodes or metrics.

Choose the occurrences before rollout. Independently cycling each
environment's queue would change the weighting of the dataset when some
environments have more recorded episodes than others. Completion order may
differ between policies, but the requested set of source records stays fixed.

The reset path must check queue eligibility before admitting an episode or
incrementing episode counters. Restoring assets without changing dispatch
and reset admission would leave replay incorrect. With several varied
objects, preserving the source environment also preserves their complete
combination of candidates.

When placement replay is requested, it must use the same selected source
record as variation replay. Read the layout from that row, or join a separate
layout file by ``(build_id, env_id, episode_in_env)``. Reject missing or
ambiguous matches. Matching by row number would be wrong when a filtered
episode file is paired with an unfiltered layout file.

This requires changes to ``PlacementLayouts`` and its reset event: the
current loader discards episode identity, the event advances its own shared
queue, and asset validation rejects heterogeneous objects. Replace that
restriction with validation against the manifest's assignments and let the
replay scheduler select the layout. Initial placement must also use a layout
compatible with the slot's restored asset; a slot with no retained episode
can receive an explicitly solved initial pose and remain inactive.

When no layout recording is supplied, placement can be solved again; that
does not restore the original initial pose. Existing behavior for unrecorded
ordinary variations can remain live sampling, but an enabled asset-selection
variation requires its manifest entry in this replay mode.

Implementation
--------------

Keep responsibilities small
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``AssetRegistry.get_asset_definition()`` is a small adapter over existing rigid
library constructors. It returns the requested registry ID and a copied native
``SpawnerCfg`` in a tuple. A constructor's ``instance_name`` does not change
that ID. Keep constructor behavior intact; no new definition class, registration
system, or broad library migration is required.

``AssetSelectionVariation`` owns the ordered copies of candidate definitions
and its sampling configuration. It emits selected names for recording, and
the builder installs the prepared native settings through generic object
assignment. The variation snapshots those settings before placement and
checks them at the final build and scene export boundaries. It does not
spawn assets or maintain geometry caches.

Extend ``Asset.add_variation()`` with a common attachment hook so selection
can bind to its host without an explicit target constructor argument. Bind a
variation instance to only one host and allow only one selection variation
per object. Binding does not sample or resolve assets. Existing variations
that receive explicit targets can retain that API during migration; adopting
the hook must preserve or validate those target references.

``Object`` remains the interface used by tasks and placement. It owns generic
asset assignment and geometry access. It accepts a default native definition
or waits for assignment during the build. It has no dependency on the
``AssetSelectionVariation`` class and does not inspect its candidates or enabled
state. Once assigned, it exposes the effective native configuration, assignment,
and geometry for each environment. No additional object class or state wrapper
is needed for this change.

``ArenaEnvBuilder`` owns construction order and validates that every object has
an assigned asset after overrides and variations. Selection resolution maps
chosen names to native indices, prepares compatible rigid-body paths, and
binds assignments before placement. The builder composes the Isaac Lab
configuration; Isaac Lab continues to own spawning and cloning.

``VariationRecorder`` owns sampled values. The episode recorder obtains the
fixed value for that episode's environment, alongside any values sampled on
reset. Future replay changes belong to the replay loader and scheduler, which
will own definition checks, restoration, and episode dispatch.

Resolve before anything consumes geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The construction order should be explicit:

1. Create a fresh environment definition, keeping shared references between
   its scene, tasks, relations, and variations.
2. Apply experiment overrides and create the build context: environment
   count, seed, and variation key. Attach recording before any draws.
3. Validate supported combinations, then apply enabled variations. Selection
   samples IDs and supplies the effective native settings and fixed assignment.
4. Validate that every object has an assigned asset. An object without a
   default and without an effective assignment fails here, before placement.
5. Prepare compatible rigid-body paths and per-environment geometry, then
   freeze the resolved asset settings.
6. Solve placement using those exact assignments.
7. Derive task sensors and compose scene, manager, and reset configurations.
8. Construct the simulation and start episodes.

The replay proposal above describes assignment restoration and a build
manifest. Neither is part of this construction path; selection with recorded
variation or placement inputs is rejected.

Keep fingerprints of the authored candidate definitions separate from
construction-time consistency checks on the final effective configs. If a
future property variation changes scale, its sampled value and final geometry
must agree without pretending the original candidate definition changed.

Changing only ``Scene.assets["pick_up_object"]`` would break references
already held by tasks and relations. Resolve the existing scene object within this build
instead. Once resolved, changes to its candidate settings or assignments
require a fresh build; otherwise cached bounds and spawned geometry can
disagree.

Use a fresh environment definition for each build, as the experiment factory
already does. Copy candidate configs when declaring them and again into the
resolved build state as needed. Do not carry bound assignments from one build
to the next. The first implementation can reject reusing an already resolved
definition in another builder; supporting concurrent builds from the same
live object graph is unnecessary for this API. This check belongs to the
definition's resolution lifetime, so a second builder cannot bypass it.
Composing a configuration and then constructing the simulation from that
configuration is the continuation of the same build, and remains supported.
A failed construction that has already resolved the definition requires a
fresh factory result.

Preserve an explicitly supplied default separately from the resolved selection
where needed for build validation and future definition checks. There may be
no default. Copy candidate names and settings together at declaration; never
reconstruct names later from a changed source configuration.

An unassigned ``Object(name)`` rejects geometry queries. An object with a
default can describe that default before variations run; it does not need to
know whether selection is enabled. The builder's ordering ensures placement
and task sensors consume the final assignment. After resolution, placement
uses the existing per-environment geometry interface. A single-box query
continues to reject a multi-asset configuration, even when every slot happens
to select the same candidate. Scene USD export separately rejects enabled,
unresolved selection rather than exporting a default as though it were selected.

Extend the variation lifecycle and recorder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The draft stack passes ``VariationBuildContext`` to build-time configuration
and adds ``sample_per_environment`` to the common variation config. Existing
variations retain explicit defaults and supported settings: shared build-time
lighting variations stay shared, and runtime variations remain per-environment.
Selection supports both settings and defaults to ``True``.

The recorder distinguishes a shared build value, a fixed build value for each
environment, and values sampled for individual episodes. It uses the variation's
declared lifecycle rather than inferring lifetime from environment IDs or
whether a live environment has been bound.

For per-environment selection, request ``num_envs`` samples with environment
IDs ``0`` through ``num_envs - 1``. For shared selection, request one sample.
Give both choice samplers the same draw interface: a sample count, an ordered
list of candidate names, and optional environment IDs. ``ChoiceSampler`` and
``SequentialChoiceSampler`` use that interface. Sequential sampling is the
selection default. Pass copied names as choices; neither sampler owns asset
definitions or scene objects.

Sequential selection starts from the first candidate for each build and
cycles in environment order.

When random sampling is selected, give each selection variation a reproducible
random stream derived from the build seed and its full key, such as
``pick_up_object.asset_selection``. Use a stable derivation rather than
Python's process-dependent hash. The choice sampler accepts an explicit Torch
generator. Adding an unrelated variation must not change the chosen fruit.
Candidate order may affect a fresh draw. Restoring recorded choices is part
of the separate replay proposal above.

Future replay work must update the loader to retain source environment and
episode identity. The current loader requires every build-time value to be
identical across all episode rows, which is correct for shared values but wrong for fixed
per-environment values. Existing homogeneous recordings without a manifest
must keep their existing replay path; they cannot be interpreted as complete
heterogeneous recordings.

Defer sensors and define composition limits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first refactor defers scene configuration in ``PickAndPlaceTask``,
``ObjectInTask``, and ``SortMultiObjectTask``. Their constructors keep scene
object references, and their configuration methods resolve contact paths
after asset preparation. This also allows tasks to hold an ``Object(name)``
whose asset has not yet been assigned.

Initially reject ``ObjectReference`` paths into an object with enabled asset
selection. A child prim that exists in a banana need not exist in an orange.
Relations to the whole scene object remain supported. Supporting candidate
specific child references or affordances is a separate design problem.

Candidate-specific scales, materials, and physics settings come directly
from native configs. Runtime mass variation can target the stable scene
object once its rigid bodies exist. General per-environment scale variation
is more involved: asset choice and scale together may require additional
effective native configurations. Until that composition is implemented,
reject unsupported geometry-changing combinations rather than mutating one
arbitrary candidate's ``spawn_cfg``.

Validate that candidates are concrete, rigid, and compatible with the task's
required sensors. Reject nested multi-asset spawners and custom bounds that
cannot be carried through the native geometry path. Preserve the stack's
physics-backend and cloning compatibility checks; a new authoring API does
not remove those simulation constraints.

Reuse the native backend work
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base implementation on the refreshed #1419 branch,
``cvolkcvolk/refactor/native-heterogeneous-spawning``, at ``0bc30cec``.
Its ancestry includes current ``main`` at ``2cc9e976`` and all the backend
work from #1415 through #1419. The refreshed builder retains the merged
variation replay integration and resolves build-time variations and asset
assignments before placement.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Existing work
     - How this design uses it
   * - `#1415 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1415>`_
     - Native configurations own asset settings. Selection copies those
       configurations instead of reconstructing assets from USD paths.
   * - `#1416 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1416>`_
     - ``ObjectGeometry`` derives bounds and meshes from the same settings
       used for spawning.
   * - `#1417 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1417>`_
     - Prepare compatible rigid-body paths when candidates need it.
       Different scales alone do not require rewritten USDs.
   * - `#1418 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1418>`_
     - Keep preview and export paths consistent with native configs.
       This is not the core selection mechanism, but the full #1419 has
       dependencies on this integration.
   * - `#1419 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1419>`_
     - Reuse fixed assignments, per-environment placement, and native
       heterogeneous spawning. Replace its assignment policy with variation
       samples or restored IDs.
   * - `#1420 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1420>`_ and
       `#1421 <https://github.com/isaac-sim/IsaacLab-Arena/pull/1421>`_
     - Their public rename and ``per_environment_objects`` authoring schema
       are not prerequisites. The direct variation API supersedes that
       proposed public surface.

This lets us remove the separate public set/assignment configuration path
and the need for a second YAML object category. The backend work also removes
Arena's need to reconstruct member settings and bake scale into special USD
copies. We retain the smaller compatibility preparation needed for genuinely
different rigid-body paths, and let Isaac Lab spawn and clone from the native
configs. There must be no second random draw inside spawning after placement
has already used an assignment.

Native spawning does not automatically solve USD scene export for a
heterogeneous object. Keep that explicit limitation until export can identify
which environment's resolved asset it is exporting. This is separate from
the recorded-placement replay work described above.

Suggested implementation sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use four new PRs after #1419. #1420 and #1421 remain separate from this
implementation branch; useful tests and validation cases can be adapted
without taking their public API changes.

The current draft stack covers the first two steps, including the native
definition API revision. Replay, graph authoring, and migration remain future work.

1. **Prepare Object and task configuration.** Move assignment state,
   per-candidate geometry caching, per-environment bounds, and common contact
   paths from ``RigidObjectSet`` into ``Object``. Derive task sensors when
   the builder requests scene configuration. Keep the existing set API
   working throughout this refactor.
2. **Add asset selection through variations.** Introduce the attachment hook,
   common ``sample_per_environment`` config, explicit build context, and
   fixed per-environment recording. Add ``AssetSelectionVariation`` with
   ``asset_candidates`` containing copied native definitions, unique-name
   validation, sequential assignment by default, optional random sampling,
   and shared values. Add the small registry adapter and let the same generic
   ``Object`` accept a default or wait for assignment. Validate all assignments
   after variations and before placement. Retain existing library constructors
   and keep ``Object`` independent of the selection class. Reject selection
   replay until the manifest and replay changes are available.
3. **Restore heterogeneous recordings.** Add the manifest, definition
   validation, preserved-environment dispatch, reset admission, and
   coordinated layout replay. This completes the Python selection and
   recording/replay behavior.
4. **Add graph authoring and migrate callers.** Add YAML and discovery, then
   migrate examples, authoring tools, and existing callers.
   A temporary ``RigidObjectSet`` adapter can translate ordered/random
   settings to an enabled selection variation. Its compatibility behavior
   needs separate review; it must not introduce an implicit default for the
   new generic ``Object`` API. Deprecation and removal happen after migration.

These steps are independently reviewable. The Python API and fixed episode
recording can be used before replay and YAML support, with explicit guards
on unsupported combinations.

Scope of the first refactor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``assets/object.py``, move the assignment tuple and the geometry cache
indexed by asset index into the base object. Move the methods that enumerate
native alternatives, bind assignments, query bounds per environment, and
validate common contact paths with that state. Continue reading configurations
from ``object_cfg.spawn``; do not introduce a second mutable source list.
A separate helper class is unnecessary for this extraction.

``assets/object_set.py`` keeps its constructor, member validation,
``random_choice`` option, and native asset preparation. It delegates geometry
and assignment handling to ``Object``. Ordinary ``Object`` construction and
its public spawn-config setter keep their current restrictions in this PR;
the variation PR introduces the internal entry point for resolving choices.

Preserve existing mutation boundaries. Assignment binding fixes indices, but
does not freeze native settings. Geometry must still refresh after a native
setting changes before the builder captures its placement snapshot. The
builder's final validation continues to reject changes that would make
placement and spawning disagree. Shared geometry code must use the object's
actual type, preserving the different geometry frames of rigid, articulated,
and static objects, as well as custom bounds on concrete objects.

In ``tasks/pick_and_place_task.py`` and ``tasks/object_in_task.py``, stop
building scene configuration in the constructor. Keep stable sensor names
and derive the sensor configurations in ``get_scene_cfg()``. In
``tasks/sorting_task.py``, also remove the constructor-time list of concrete
contact sensor configurations and derive them from the stored object pairs.
An early call to ``get_scene_cfg()`` must not permanently cache old paths.

``CompositeTaskBase.get_scene_cfg()`` should collect each child's scene
configuration once and reuse those results for duplicate checking and
composition. The existing builder already requests these configurations
after variations, assignment, and placement, so this refactor needs no new
task lifecycle hook.

Keep ``ObjectReference`` behavior unchanged in this first PR. The variation
PR must reject references into selected parents after experiment overrides
have been applied, before asset selection is realized. A constructor-only
check would miss a variation enabled later through configuration.

Validate this refactor with the existing object configuration, geometry,
assignment, and native heterogeneous-scene tests. Add focused regression
coverage that changes an object's and destination's native body paths after
task construction, then verifies that each task's requested sensor config
uses those paths. Repeat the query after another change to catch stale
caches. Use small local USD fixtures, and retain coverage for ordinary
objects, single-candidate sets, and deformable pickups without contact sensors.

What must be demonstrated
^^^^^^^^^^^^^^^^^^^^^^^^^

Use a small rigid-object scene to verify the whole path:

* ``Object(name, asset=definition)`` uses its default when selection is absent
  or disabled. ``Object(name)`` requires an enabled assignment variation and
  otherwise fails before placement, without a first-candidate fallback.
* The registry adapter snapshots native settings without transferring library
  behavior, poses, relations, or variations. Both construction forms use the
  same generic object class and geometry interface.
* Sequential assignment is the default; selecting the random sampler preserves
  the existing optional behavior. Resets never change either assignment.
* ``sample_per_environment=True`` produces one value per environment and
  ``False`` produces one shared value. Existing variations retain their
  defaults and reject unsupported settings.
* Native scale and physics settings survive selection. Placement bounds and
  contact sensors refer to the actual asset spawned in each environment.
* Fresh builds resolve independently. An unrelated variation does not alter
  selection, and reusing stale resolved state fails clearly.
* Candidate names distinguish differently configured definitions from one
  registered asset. Duplicate names fail at declaration. Later edits to a
  supplied native configuration do not change the copied settings. Reordering
  candidates changes fresh sequential assignment without changing their IDs.

Future replay work must also demonstrate:

* Missing names and changed definitions fail before simulation starts.
  Reordering candidates preserves recorded identities.
* A recording with unused slots still has complete assignments. Filtered
  replay preserves those assignments and runs only retained episodes.
* Policies that finish episodes in different orders receive the same selected
  source records. Explicit cyclic budgets preserve file weighting, including
  when some environment queues run out earlier than others.
* Placement and variation replay use the same source episode. Existing shared
  build-time variations and homogeneous replay remain compatible.

Questions for the design review
-------------------------------

Future review should settle the supported configuration types for definition
fingerprints and the YAML representation of defaults and candidate definitions.
It should also decide the migration path and how long to retain
``RigidObjectSet``. Those decisions do not require a new library model or an
affordance rewrite in this change.

Runtime asset replacement and replay with a different environment count are
future work. The latter would require dispatch by the complete combination
of selected assets, rather than by source environment ID. Neither is needed
to make the first direct variation API useful.

Background
----------

The implementation plan was checked against ``main`` at ``2cc9e976`` and
the refreshed backend stack through #1419 at ``0bc30cec``. The existing
`variation record/reload proposal
<https://docs.google.com/document/d/1j_QLJ-PQp_H_dNLLL-XtFpLx16dWNhYxXXLyYlug1uM/edit?tab=t.rfdoasfvr500>`_
motivates recording object identity and proposes one shared choice per
rebuild as a simpler starting point. This design keeps that option and also
supports the agreed goal of different fixed choices within the same build.
