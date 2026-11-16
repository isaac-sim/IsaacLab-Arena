Release Notes
=============

.. toctree::
   :hidden:

   migration_guide_v0_4

v0.4.0
------

This release introduces the following features:

- Agentic authoring of evaluation environments with coding agents.
- Multimodal vision-language models (VLMs) as robot policies.
- Reproducible evaluations through seeded, recorded, and replayed variations and placements.
- An update to the Isaac Lab 3.0 release, Isaac Sim 6.2, and Newton 1.6.

And the following new workflows:

- Agentic authoring of workbench cleanup and kitchen pick-and-place environments.
- Newton-based evaluation of contact-rich gear insertion and deformable pick-and-place.
- VLM agent policy evaluation on DROID pick-and-place.

To update environments, scripts, and output parsers from v0.3, see the
:doc:`migration guide <migration_guide_v0_4>`.

Details below:

**Features and improvements**

- **Agentic authoring:** Added an environment-authoring skill and agent guidance, an object-control
  runner that checks task predicates through physics without a robot policy, a scene inspector with
  USD physics metadata, and workbench cleanup and KitchenBench authoring workflows (#1400, #1563,
  #1607, #1615, #1652, #1653, #1660).
- **VLM policies:** Added ``VLMGoalPolicy`` and ``VLMChunkPolicy`` with
  robot-specific observation and action adapters, a shared multimodal inference backend, and
  Experiment Runner support with DROID pick-and-place and gear insertion examples (#1414, #1479,
  #1482, #1521, #1575, #1576).
- **Newton workflows:** Added a Newton gear insertion environment with DROID teleoperation,
  recording, and replay, and Newton DROID deformable pick-and-place. Newton runs now abort on
  physics buffer overflow (#1385, #1409, #1441).
- **Reproducible variations and placement:** Seeded each variation from ``--seed``, and added
  replay of recorded variation values and placement layouts across episodes, unified around
  ``PlacementSample``. Added an ``asset_selection`` variation that spawns a different object asset
  per rebuild (#1393, #1394, #1427, #1429, #1452, #1458, #1467, #1517).
- **Placement relations and diagnostics:** Added the ``AlignWith`` relation and
  ``OrientedBoundingBox`` collision checks, improved ``NextTo`` seeding, and allowed ``ClutterOn``
  supports to move through their own relations. Placement now reports per-check validation counts,
  written to each Run's output directory, and is measured by placement-success Experiments
  (#1389, #1477, #1480, #1522, #1556, #1560, #1597, #1602, #1623).
- **Assets and object references:** Added nested ``ObjectReference`` chains,
  ``ArticulationObjectReference``, and placement of references to articulation links at their
  initial joint positions. Added ``UsdSource`` acquisition and search for SimReady Central assets
  (#1513, #1572, #1617, #1621).
- **Experiments and KPIs:** Experiments now have a name and record Arena git provenance. Added
  ``ArenaExperimentResult.from_experiment_output()`` to read results back, InfluxDB KPI exporters,
  a Python API for OSMO Experiment submission, and tracked events that record recurring
  occurrences such as dropped objects (#1457, #1538, #1540, #1585, #1586, #1587, #1588).
- **Cameras:** Added ``camera_resolution_mode``, which is off by default. When activated, its
  ``policy_default`` mode shrinks embodiment cameras to fit the policy's image size, keeping each
  camera's aspect ratio. With pi0.5 on DROID, this cuts camera memory from about 1 GB to under
  40 MB per environment, so many more camera-based environments fit on one GPU (#1511).
- **Task libraries and benchmarks:** Added eleven KitchenBench environments for knob turning,
  multi-object and multi-level pick-and-place, spatial placement, stacking, and microwave door
  opening. Aligned RoboLab environments with RoboLab, and added PerfLab benchmark Experiments
  (#1487, #1518, #1595).

**Documentation**

- Added a What's New section on agentic authoring, Newton workflows, and VLM policies, and aligned
  the homepage, overview, and README with it (#1579, #1600, #1609).
- Added new example workflows:

  - **Agentic authoring, workbench:** Prompt a coding agent to author a workbench cleanup task,
    and run a force-based reference demonstration (#1652, #1653).
  - **Agentic authoring, kitchen:** Prompt a coding agent to author a two-bowl pick-and-place task
    in a KitchenBench kitchen and review the result (#1660, #1679).
  - **Newton gear insertion:** Preview, teleoperate, record, and replay a DROID gear insertion
    task, and evaluate a VLM agent policy on it (#1510, #1575, #1633).
  - **Newton deformable pick-and-place:** Preview the task on the Newton or PhysX deformable
    backend, teleoperate, and evaluate OpenPI in closed loop (#1441, #1634).
  - **VLM agent policy:** Evaluate goal, chunk, and step VLM policies on DROID pick-and-place
    (#1551, #1675).
  - **Recorded variation replay:** Replay recorded variation values from one or more result files
    (#1429).

- Added object-control and placement diagnostics guides, a generated relations reference, and a
  community contribution guide. Restructured the policy concept docs into four policy families
  (#1498, #1602, #1607, #1610, #1641).
- Removed the legacy agentic environment generation, Dexsuite Newton RL, and GR1 open-microwave
  tutorial pages (#1619, #1666, #1667).

**Dependencies**

- Updated Arena to the Isaac Lab 3.0.0 release, Isaac Sim 6.2, and Newton 1.6.1, with a
  ``uv``-based Docker installation (#1481).
- Included cuRobo in every native ``uv`` and Docker install, replacing the separate cuRobo image,
  and upgraded the IK reachability check from cuRobo v1 to v2. The check now runs by default for
  embodiments with a cuRobo config, such as DROID (#1550).

**Infrastructure and CI**

- Baked the Arena git commit and branch into the Docker image (#1539).
- Improved CI caching and skipped runtime CI for docs-only PRs (#1571, #1624).

**Bug fixes**

- **Placement correctness:** Decorrelated candidate seeds across placement seeds, fixed the pool
  fallback and yawed mesh bounds, and failed early on relations on mesh-collision backgrounds
  (#1523, #1574, #1598, #1614).
- **Recording and resets:** Fixed demo recording with variations and placement, and reset
  articulated objects to their initial joint positions with zero joint velocity (#1435, #1502,
  #1573).
- **Rendering and interop:** Fixed camera initialization in Isaac Lab interop, kept Fabric
  enabled in the environment runner for stable CPU PhysX scenes, and fixed camera intrinsics
  variation so per-environment intrinsics appear in rendered images (#1508, #1554, #1613).
- **OSMO:** Fixed timings filename collisions and Run timeouts in the Experiment output collector
  (#1426).

**Limitations**

- Installation from a published Python package is not yet supported; use the native ``uv`` source workflow or Docker.


v0.3.1
-------

This release introduces:

- Updates the simulation stack with expanded Newton support for objects and robots.
- Adds simulation setup and placement parameter overrides through environment YAML.
- Adds placement recording and replay, with offline clutter generation and validation.

Details below:

**Features and improvements**

- **Object placement and clutter:** Added recording and replay of settled placement
  layouts, ``ClutterOn`` release poses, and offline clutter settling and validation.
  Added footprint-overlap constraints for ``On`` relations and made
  ``OnTreeInitializer`` the default placement initializer (#1265, #1291, #1308, #1318,
  #1319, #1320, #1369).
- **Task completion and progress:** Unified success checks and progress tracking
  around ``CompletionCriteria``. Added consecutive-step requirements, one progress
  update per control step, and episode-end streak counts and first-completion steps
  in Experiment Runner JSON and HTML reports (#1255, #1304, #1305, #1307, #1316, #1382).
- **Runtime scene access consolidation:** Added ``ArenaWorld`` as the read-only runtime
  scene facade on ``ManagerBasedEnv``. Migrated task predicates, rewards, terminations,
  observations, and metrics to query through it (#1173, #1187, #1195, #1197).
- **Environment YAML extensions:** Added ``env_cfg_override`` to configure
  simulation timestep, control decimation, solver, and collision settings directly
  in environment definitions. Select the backend with ``default_physics_backend``
  or ``--presets``; overrides tune the selected backend. Added YAML placer overrides,
  graph asset parameters and initial poses (#1266, #1302, #1363).
- **Evaluation and recording:** Added rollout timing and trajectory extraction. Run
  exactly the requested number of evaluation episodes (#1198, #1254, #1371).
- **Embodiments and physics backends:** Enabled Newton-backed DROID control and
  backend hooks on embodiments. Added an embodiment-owned gripper interface and
  environment-specific spawn physics settings (#1189, #1284, #1293).
- **Heterogeneous object sets:** Added support for multiple heterogeneous object sets
  in one scene, including contact-sensor path validation and a multi-object
  pick-and-place example (#1381).
- **Deformables and cables:** Added PhysX deformable pick-and-place assets and tasks,
  and a cable asset class (#1191, #1204, #1276, #1284).
- **New tasks and predicates:** Added ``ObjectInTask`` for placing and settling objects
  in containers. Added shared gripper release, withdrawal, and ``gripper_not_grasping``
  predicates for insertion and manipulation tasks (#1276, #1370, #1390).
- **New variation:** Added ``ObjectDisappearVariation``, a run-time Bernoulli
  distractor-disappearance variation on rigid objects, disabled by default and
  configurable per object from the CLI or experiment YAML (#1268).

**Documentation**

- Added or expanded guides for offline placement recording, clutter generation,
  physics configuration, embodiments, YAML overrides, and consecutive-step summaries
  (#1266, #1284, #1308, #1320, #1330, #1372, #1382).

**Infrastructure and CI**

- Split Docker builds into stages and setup scripts, and added premerge image builds
  using remote NGC caches (#1280, #1282).

**Bug fixes**

- **Multi-rebuild evaluation:** Isaac Sim 6.1 fixes scene displacement after
  environment rebuild and removes the CPU physics fallback during multi-rebuild
  evaluation (#1237, #1174, #1192).
- **Physics presets:** Fixed Arena preset application so preset Newton settings no
  longer override user-configured Newton settings (#1262, #1266).
- **Task correctness:** Refined settling and spatial success checks, restored Arena
  success tracking during Mimic generation, and required complete subtask definitions
  in generated environments (#1260, #1337, #1368).
- **Placement and physics:** Fixed collision bounds for ``RotateAroundSolution`` yaw
  and skipped velocity resets for kinematic nested background rigid bodies (#1272, #1339).
- **Cameras and startup:** Fixed camera-extrinsics variation under Newton, skipped
  non-RGB observations during video recording, and corrected interactive runner USD
  startup order (#1196, #1312, #1327).

**API migration**

- Replace ``ProgressObjective`` with ``CompletionCriteria`` and predicate groups with
  predicate sequences. Return a ``TaskTerminationCfg`` from ``get_termination_cfg()``
  and declare success through its ``success`` field (#1255, #1316).
- Replace ``SequentialTaskBase`` with ``CompositeTaskBase`` using
  ``subtasks_are_sequential=True`` for ordered subtasks. Pass a flat subtask list;
  nested composite tasks are not supported (#1255).
- Use ``TrueForConsecutiveStepsCfg`` around an instantaneous predicate to require a
  continuous streak; the progress runner owns and resets the counters (#1305).
- Update USD helper imports to concrete submodules such as
  ``isaaclab_arena.utils.usd.helpers`` and ``isaaclab_arena.utils.usd.rigid_bodies``.
  Import ``ObjectType`` from ``isaaclab_arena.assets.object_type`` (#1218, #1292).

**What's Next**

Future releases will focus on:

- Newton-based example workflows for contact-rich and deformable tasks.
- Reference examples for LLMs as policies.
- Improve Arena agentic readiness for complex long-horizon task generation.

v0.3.0
------

This release introduces:

- Prompt-first environment generation.
- Controlled randomization through variations.
- Sensitivity analysis.
- Multi-node evaluations.
- Expanded policy integrations.
- Task libraries, including RoboLab-style tasks and our own Kitchen Benchmark for tasks in
  realistic kitchens.

Details below:

**Features and improvements**

- **Agentic environment generation:** Added ``EnvironmentGenerationAgent`` for creating environments from
  natural-language prompts, with intent-to-graph conversion, SimReady asset search, composite-task
  support, configurable inference endpoints, and an interactive review GUI with simulation
  snapshots and relation-solver previews (#718, #770, #803, #804, #805, #868, #982, #1050).
- **Typed environment definitions:** Added Pydantic-based ``ArenaEnvGraphSpec`` YAML parsing and CLI
  bring-up, typed environment and policy configurations, a reusable environment factory, and a
  unified path for loading registered Python environments or graph YAML files
  (#690, #749, #757, #808, #858, #860, #862, #863).
- **Isaac Lab 3.0 workflow interoperation:** Updated the Isaac Lab submodule and moved
  demonstration recording, replay, and teleoperation to Isaac Lab's scripts through Arena's
  external environment registration callback (#960, #994, #1022).
- **Experiments and evaluation reports:** Replaced legacy evaluation jobs with typed YAML
  Experiments and Runs, including environment rebuilds, chunked and multi-node dispatch,
  combined JSON results, an experiment viewer, grouped success-rate plots, and generated
  evaluation reports (#751, #772, #802, #867, #873, #896, #1042, #1056, #1070).
- **Variations and sensitivity analysis:** Added Hydra-configurable variations with in-memory
  recording for lighting, camera intrinsics, and object mass, plus an MNPE/NPE sensitivity-analysis
  workflow, sample dataset, and plotting tools (#729, #746, #755, #775, #789, #913, #916, #1055).
- **Object placement and validation:** Added heterogeneous and deterministic placement,
  mesh-based non-collision constraints, ``NotNextTo`` and ``FaceTo`` relations, unified placement
  APIs, build-time physics settling, and pluggable validation checks (#676, #679, #732, #769,
  #771, #809, #888, #927).
- **Robot reachability validation:** Added a build-time, simulation-free cuRobo IK gate for
  task-relevant object layouts, including Droid and Franka embodiment placement, collision
  checking, and optional debug visualization (#914, #947, #954, #984, #1014).
- **Policy integrations:** Added OpenPI support for DROID and Pi0.5, GR00T remote closed-loop
  policies with pluggable action schedulers, Cosmos and DreamZero policies, and OSMO workflows
  for parallel and multi-policy evaluation (#595, #655, #658, #663, #688, #823, #843, #963,
  #991, #1005).
- **Task libraries:** Added the RoboLab catalog and more than 15 pick-and-place environments,
  plus Lightwheel and Replicator kitchen environments for pick-and-place, opening doors and
  microwaves, pressing buttons, and turning knobs (#845, #857, #962, #965, #966, #1048, #1149,
  #1155).
- **Progress tracking:** Added fine-grained subtask state, episode recording, order-independent
  composite tasks, unified success checks, and normalized progress scoring (#677, #758, #810,
  #822, #840, #1137).

**Documentation**

- **Agentic generation workflows:** Added concept, validation, tabletop, kitchen, GUI,
  reachability, and collision-handling documentation for prompt-first environment creation
  (#844, #1025, #1028, #1033, #1046, #1064, #1076, #1085, #1135).
- **Evaluation workflows:** Added documentation for Experiments and Runs, generated reports,
  multi-node execution, variations, sensitivity analysis, environment seeds, and
  ``RigidObjectSet`` (#1030, #1045, #1051, #1053, #1063, #1068, #1072).
- **Website and catalogs:** Added the Kitchen Benchmark and RoboLab catalogs, redesigned the
  overview and motivation pages, and updated the release documentation for Arena 0.3
  (#1041, #1048, #1089, #1126, #1217, #1222).

**Infrastructure and CI**

- **Native installation:** Added a fully source-based ``uv`` installation path with lightweight
  OpenPI and GR00T clients, runtime package configuration, and dedicated native-``uv`` CI
  validation (#908, #934, #938, #944, #1084, #1092, #1101).
- **Docker workflows:** Added per-clone container naming, host-user command execution, optional
  container suffixes, and improved Docker support for agent workflows (#643, #760, #762, #763).
- **CI coverage:** Added all-environment smoke testing, a GR00T closed-loop end-to-end job,
  camera test coverage, and improved Isaac Sim shader caching and subprocess isolation
  (#617, #618, #644, #675, #950).

**Assets and tests**

- **Kitchen assets:** Added Replicator kitchen examples, publicly hosted USD-randomizer kitchens,
  Lightwheel kitchen registration, mesh-placement examples, and kitchen task configurations
  (#962, #965, #1088, #1090).
- **RoboLab assets:** Added the RoboLab catalog, missing object assets, agent-generated
  pick-and-place environments, and OpenPI-versus-Cosmos experiment configurations
  (#815, #817, #820, #845, #857, #1020).

**Bug fixes**

- **Placement correctness:** Fixed relation-solver bounding-box scaling, rotation-around
  placement failures, IK reachability for relation-placed robots, and geometric pick-and-place
  success checks (#816, #921, #1081, #1132).
- **Rendering and cameras:** Fixed corrupt kitchen rendering, Droid camera quaternion ordering,
  review-GUI viewport setup, and viewpoint camera poses in parallel environments
  (#673, #941, #1043, #1223).
- **Evaluation correctness:** Fixed normalized progress scores, missing-video report handling,
  timeout termination semantics, and external environment registration for Isaac Lab workflows
  (#865, #994, #1017, #1137).
- **Background physics:** Fixed kitchen placement collisions, relation anchors in background
  collision meshes, and physics reset behavior for kitchen and other backgrounds
  (#988, #1003, #1102, #1180).

**Limitations**

- Installation from a published Python package is not yet supported; use the native ``uv`` source workflow or Docker.

**What's Next**

Future releases will focus on:

- Enhanced Newton support for benchmarks involving contact-rich insertion, cables, and deformables.
- Prompt-to-scene, robot, and task workflows.
- Improved performance for multi-node evaluation.


v0.2.0
------

This release introduces major new capabilities including Isaac Lab 3.0 (Newton) support,
a sequential task chaining framework, semantic object placement, teleoperation, GR00T N1.6
and DROID integration, RL workflows, and a large set of new tasks and embodiments.

**Features and improvements**

- **Isaac Lab 3.0 (Newton) upgrade:** Updated the framework to Isaac Lab 3.0 (Newton),
  including an updated interop layer (#464, #533).
- **Sequential task framework:** Added ``SequentialTaskBase`` class for chaining atomic
  skills into long-horizon tasks, with mimic and metrics support, user-specifiable
  final subtask success states, and an example that puts an object into a microwave and
  closes the door (#289, #323, #337, #365).
- **Semantic object placement:** Added a differentiable relation-based object placement
  solver supporting ``On``, ``NextTo``, ``AtPosition``, and ``PositionLimits`` relations,
  multiple anchors, ``RotateAroundSolution``, and full integration with ``ArenaEnvBuilder``
  and ``ObjectPlacer`` (#328, #354, #358, #362, #387, #574).
- **Teleoperation:** Added teleoperation support for G1 loco-manipulation and GR1 using
  Quest XR hand-tracking and CloudXR; updated to IsaacTeleop v1.1 (#286, #350, #577, #605).
- **GR00T N1.6 and DROID integration:** Upgraded GR00T to N1.6 and added DROID dataset
  support, local inference pipeline, and language instruction support for closed-loop
  evaluation (#334, #416, #418, #420, #519).
- **New tasks, embodiments, and evaluation capabilities:** Added Sorting, FactoryAssembly,
  TurnKnob, CloseDoor, and AdjustPose atomic tasks; Galbot and Agibot A2D embodiments;
  G1 WBC-AGILE end-to-end velocity policy; RSL-RL policy evaluation; distributed
  multi-GPU policy runner; multi-task evaluation job runner (#371, #285, #315,
  #295, #305, #391, #292, #489, #333, #411, #277, #445, #394).

**Documentation**

- **README and Getting Started overhaul:** Comprehensive README rewrite and full
  reorganization of the Getting Started section and navigation structure (#480, #505).
- **Concepts and placement docs:** Added a humanized concepts page and a dedicated
  object placement documentation page (#526, #511).
- **RL and evaluation workflows:** Added RL workflow docs, policy evaluation section,
  Newton evaluation example from IsaacLab DexSuite, and GTC DLI workflow docs
  (#363, #451, #536, #390).
- **External repository and advanced usage:** Added a dedicated page for using Arena
  from an external repository, an advanced custom task example, GR00T closed-loop
  docs, and DROID usage instructions (#518, #550, #519).

**Infrastructure and CI**

- **Docker and dependency improvements:** Moved Python dependencies from the Dockerfile
  to ``setup.py``; pinned Newton mujoco version; fixed docker pipx stall; added missing
  Arena package to NGC docker (#535, #629, #640, #578).
- **Repository governance:** Added CODEOWNERS, issue templates (brought over from
  IsaacLab), ``SECURITY.md``, ``AGENTS.md``, and ``CLAUDE.md`` (#583, #584, #585,
  #586, #492, #454).

**Assets and tests**

- **RoboLab objects and HDR library:** Added the RoboLab asset library and HDR lighting
  support for use in robolab scenes (#429, #428, #431).
- **New example environments:** Added GR1 DLI environment, G1 AGILE tabletop
  environment, and a Rubik's cube pick-and-place environment (#385, #562, #421).
- **USD asset paths:** Updated object library paths to use the ``ISAAC_NUCLEUS_DIR``
  prefix and updated USDs to reference the Isaac Sim 6.0 staging bucket (#291, #621).

**Bug fixes**

- **Sequential task metrics:** Fixed subtask success rate metric and
  ``desired_subtask_success_state`` check in sequential task evaluation (#405, #410).
- **RSL-RL evaluation metrics:** Fixed an extra episode appearing in policy evaluation
  metrics when using RSL-RL (#530).
- **Object placement correctness:** Fixed bounding box rotation in world frame, rejection
  of overlapping placements by ``ObjectPlacer``, and initialization of ``On``-relation
  objects within their parent's footprint (#400, #439, #538).
- **Recorder dataset filename collision:** Fixed filename collisions when multiple
  recorders write concurrently to a shared ``/tmp`` directory (#469).
- **Metrics serialization:** Sanitized NumPy types in metrics output to prevent
  serialization errors (#602).


v0.1.1
------

This release includes bug fixes, documentation improvements, CI and infrastructure
updates, and several API and workflow enhancements over v0.1.0.

**Features and improvements**

- **Object configuration:** Object configuration is now created as soon as an asset is
  called, so users can edit object properties before a scene is created (#239).
- **Scene export:** Added support for saving a scene to a flattened USD file (#237).
  Scene export now correctly handles double-precision poses and adds contact reporters
  when exporting rigid objects (#242).
- **Parallel environment evaluation:** Enabled parallel environment evaluation for
  GR00T policy runner, with documentation for closed-loop GR00T workflows (#231, #236).
- **Episode length:** Increased episode length for loco-manipulation to support
  rollout through box drop (#235).
- **Microwave example:** Increased reset openness for the microwave example (#311).

**Bug fixes**

- **Reference object poses:** Fixed reference object poses so they correctly account
  for the parent object’s initial pose; poses are now relative and composed at compile
  time (#232).
- **IsaacLab-to-stage path conversion:** Fixed a bug when the asset name appeared twice
  in the prim path (replaced both instances instead of one) (#241).
- **qpsolvers:** Patched breakage with Isaac Lab 2.3 due to ``qpsolvers`` upgrade by
  pinning to 4.8.1 (#252).
- **Parallel eval:** Removed comments that were breaking the parallel eval run
  commands (#262).

**Documentation**

- **Multi-versioned docs:** Documentation is now versioned so users can read docs that
  match their release (#272, #300).
- **Links and structure:** Updated README docs link to the public location (#270),
  corrected doc pointers (#301), and added release warnings (#303).
- **Installation:** Private Omniverse/Nucleus access is described on a separate page
  to clarify it is not required for normal installation (#261).

**Infrastructure and CI**

- **Runners:** Release 0.1.1 CI runners moved from local (Zurich) to AWS (#433).
- **CI workflow:** Added YAML anchors to reduce repetition in the CI workflow (#245).
- **Contribution guide:** Added signoff requirements for external contributions (#238).
- **Docker:** Fixed Dockerfile pip usage and added SSL certificate support for
  Lightwheel SDK (#449).
- **Tests:** Finetuned GR00T locomanip model is now generated on the fly in tests
  instead of mounting a pre-finetuned models directory, improving public CI
  compatibility and testing the fine-tuning pipeline (#247).

**Assets and tests**

- **G1 WBC:** Updated G1 WBC embodiment file paths to use S3 (#251).
- **Test assets:** Removed internal or custom-only assets from tests: custom cracker
  box (#234), custom USD in ObjectReference test (#240), internal asset from USD
  utils test (#244). ObjectReference test now composes USD on the fly via scene
  export (#240).


v0.1.0
------

This initial release of Isaac Lab Arena delivers the first version of the
composable task definition API.
Also included are example workflows for static manipulation tasks and loco-manipulation
tasks including GR00T GN1.5 finetuning and evaluation.

Key features of this release include:

- **Composable Task Definition:** Base-class definition for ``Task``, ``Embodiment``, and ``Scene``
  that can be subclassed to create new tasks, embodiments, and scenes.
  ``ArenaEnvBuilder`` for converting ``Scene``, ``Embodiment``, and ``Task`` into an
  Isaac Lab runnable environment.
- **Metrics:** Mechanism for adding task-specific metrics which are reported during evaluation.
- **Isaac Lab Mimic Integration:** Integration with Isaac Lab Mimic to automatically generate Mimic definitions for
  available tasks.
- **Example Workflows:** Two example workflows for static manipulation tasks and loco-manipulation tasks.
- **GR00T GN1.5 Integration:** Integration with GR00T GN1.5 including a example workflows for finetuning and evaluating
  the model on the static and loco-manipulation workflows.

Known limitations:

- **Number of Environments/Tasks:** This initial is intended to validation the composable task
  definition API, and comes with a limited set of tasks and workflows.
- **Loco-manipulation GR00T GN1.5 finetuning:** GR00T GN1.5 finetuning for loco-manipulation
  requires a large amount of GPU resources. (Note that static manipulation finetuning can be
  performed on a single GPU.)
