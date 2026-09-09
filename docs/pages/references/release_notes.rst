Release Notes
=============

Isaac Lab Arena
---------------

Isaac Lab-Arena focuses on adding essential features needed for creation and
execution of large-scale task libraries with complex long-horizon tasks.

**Key Features**

- **LEGO-like Composable Environments** — Mix and match scenes,  embodiments, and tasks independently
- **On-the-fly Assembly** — Environments are built at runtime; no duplicate config files to maintain.
- **New Sequential Task Chaining** — Chain atomic skills (e.g. Pick + Walk + Place + …) to create complex long-horizon tasks.
- **New Natural Language Object Placement** — Define scene layouts using semantic relationships
  like "on" or "next to", instead of manually specified coordinates.
- **Integrated Evaluation** — Extensible metrics and evaluation pipelines for policy benchmarking
- **New Large-scale Parallel Evaluations with Heterogeneous Objects** — Evaluate policy on multiple parallel
  environments, each with different objects, to maximize evaluation throughput.
- **New RL Workflow Support and Seamless Interoperation with Isaac Lab** — Plug Isaac Lab-Arena environments
  into Isaac Lab workflows for Reinforcement learning and Data generation for imitation learning.


**Ecosystem**
Isaac Lab-Arena is part of a growing ecosystem of tools and benchmarks. NVIDIA and partners are building
industrial and academic benchmarks on the unified Isaac Lab-Arena core, so you can reuse building blocks
(tasks, scenes, metrics, and datasets) for your custom evaluations.

- `Lightwheel RoboFinals <https://lightwheel.ai/robofinals>`_ — High-fidelity industrial benchmarks.
- `Lightwheel RoboCasa Tasks <https://github.com/LightwheelAI/LW-BenchHub>`_ — 138+ open-source tasks,
  50 datasets per task, 7+ robots.
- `Lightwheel LIBERO Tasks <https://github.com/LightwheelAI/LW-BenchHub>`_ — Adapted LIBERO benchmarks.
- `RoboTwin 2.0 <https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena>`_ — Extended simulation
  benchmarks using Arena; `Arxiv <https://arxiv.org/abs/2603.01229>`_.
- `LeRobot Environment Hub <https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot>`_ — Share
  and discover Arena environments on Hugging Face.
- `Isaac for Healthcare RHEO Workflows <https://github.com/isaac-for-healthcare/i4h-workflows/tree/main/workflows/rheo>`_ — Healthcare
  robotics benchmark workflows.

NIST Board 1, NVIDIA Isaac GR00T Industrial Benchmarks, NVIDIA DexBench, NVIDIA RoboLab, and more benchmarks
are coming soon.

**Collaboration**

Isaac Lab-Arena is being developed as an open-source, shared evaluation framework that the community can
collectively enhance and expand. We invite you to try Isaac Lab-Arena 0.3 Alpha, share feedback, and help
shape its future. In Alpha stage, development velocity is high and core features/APIs are evolving. Your
input at this stage is especially valuable.

**What's Next**

Future releases will focus on:

- Enhanced Newton support for benchmarks involving contact-rich insertion, cables, and deformables.
- Prompt-to-scene, robot, and task workflows.
- Improved performance for multi-node evaluation.

**Limitations**

- Installation from a published Python package is not yet supported; use the native ``uv`` source workflow or Docker.
- Performance is not yet hardened for production-scale workloads in Alpha stage.


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
