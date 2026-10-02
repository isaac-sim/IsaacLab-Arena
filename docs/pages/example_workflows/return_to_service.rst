.. _return-to-service:

Return to Service: Vacuum Refurbishment
=======================================

``return_to_service`` is a fixed-base DROID benchmark for servicing and
repacking a returned cordless vacuum. The robot must diagnose faults, isolate
the battery, empty and repair the airway, test the assembled appliance, and
pack a complete kit. Passive fixtures support the parts so one parallel-jaw
gripper can perform the work within the tabletop workspace.

The next action depends on evidence. A weak battery can mask an airflow
problem; replacing it does not establish that the vacuum is ready. Opening
the airway after a successful test requires another test. Working original
components must be retained. These dependencies make the task longer and
more demanding than a sequence of independent placements.

.. figure:: ../../images/return_to_service_workcell.png
   :alt: DROID overlooking the vacuum, test instruments, service trays, spare parts, and carrying case.
   :width: 100%

   Current Blender-authored assets in DROID's external camera. This frame is
   from a zero-action smoke test in the local compatibility runtime; the
   episode ended at its timeout without task success.

.. note::

   This example supplies a benchmark, verification tests, and runnable
   zero-action experiments. It does not supply a solving policy. The motion
   excerpts below illustrate individual operations performed by a development
   IK controller; they are not successful complete episodes or policy baselines.


Complete the work order
-----------------------

All returns share the same public instructions. The motor is serviceable,
provided batteries are precharged, and clogged filters are routed for cleaning
outside this episode. A typical servicing sequence is:

#. Disconnect the original battery and load-test it in the battery tester.
   Keep a passing original. If it fails, put it in the battery-service tray
   and test a compatible replacement before using it.
#. Release the collection cup with the battery disconnected. Empty its debris
   into the waste tray and remove any inlet obstruction.
#. Inspect and service the airway. Replace a clogged filter with a compatible
   spare, routing the original to the filter-service tray. Preserve a healthy
   original filter.
#. Reassemble the filter and cup, then connect the tested battery. Seat the
   removable airflow adapter between the vacuum inlet and the instrument port.
#. Press the airflow-test button and wait for a valid result. A failure
   requires further diagnosis and repair. A passing test certifies the actual
   battery, filter, and assembled airway used in that test.
#. Remove and park the adapter. Disconnect the same tested battery for
   shipping. Pack the vacuum, battery in its separate compartment, crevice
   tool, and brush tool in the carrying case; close and latch the case.
#. Leave debris and removed components in their designated trays, restore
   unused stock, and release all objects at rest.

Removing the battery for packing preserves the airflow certificate when the
same tested battery is packed and the airway remains intact. Substituting a
battery or changing the airway invalidates it. Accessing the cup or filter
while the battery is connected prevents success for that episode.

There are three independent fault conditions and eight assignments:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Scenario
     - Weak battery
     - Clogged filter
     - Inlet obstruction
   * - ``healthy``
     - No
     - No
     - No
   * - ``battery``
     - Yes
     - No
     - No
   * - ``filter``
     - No
     - Yes
     - No
   * - ``obstruction``
     - No
     - No
     - Yes
   * - ``battery_filter``
     - Yes
     - Yes
     - No
   * - ``battery_obstruction``
     - Yes
     - No
     - Yes
   * - ``filter_obstruction``
     - No
     - Yes
     - Yes
   * - ``combined``
     - Yes
     - Yes
     - Yes

Even a healthy return contains collection-cup debris and requires testing and
packing. ``scenarios`` assigns conditions cyclically across parallel
environments; resets repeat that assignment. The full example evaluates each
assignment in a separate named Run. Layouts and faults are not randomized on
reset.

These GIFs show development IK excerpts from a local compatibility run and
an earlier asset build. The source episode did not complete the full task;
it later stopped at airflow-adapter insertion. The controller used to produce
this footage is not part of the benchmark implementation.

.. figure:: ../../images/return_to_service_battery_load_test.gif
   :alt: DROID pressing the battery tester button after seating a battery.
   :width: 80%

   Triggering the battery load test.

.. figure:: ../../images/return_to_service_cup_removal.gif
   :alt: DROID grasping and lifting the detached collection cup.
   :width: 80%

   Detaching and lifting the collection cup.

.. figure:: ../../images/return_to_service_cup_emptying.gif
   :alt: DROID rotating the collection cup above the waste tray.
   :width: 80%

   Tilting the collection cup over the waste bin.


Generate the Blender assets
---------------------------

All nonrobot geometry, collision proxies, materials, texture maps, and labels
come from ``isaaclab_arena_environments/return_to_service/asset_source/``.
The original assets were built through Blender MCP. The same source entry
point supports reproducible local generation with Blender 4.5 LTS and its
bundled NumPy and USD modules:

.. code-block:: bash

   blender --background --python \
     isaaclab_arena_environments/return_to_service/asset_source/build_assets.py \
     -- --output-dir "$HOME/.cache/isaaclab_arena/return_to_service/assets" --render

The source README includes the Blender MCP invocation. Export validation
checks topology, UVs, textures, collision shapes, assembly clearances, and
articulated sweeps. Generated USDs, textures, the manifest, and the standalone
``.blend`` file belong outside the repository. Runtime physics layers are
cached separately and continue to reference their source assets.

The default asset root is
``~/.cache/isaaclab_arena/return_to_service/assets`` for the user running Arena.
The Docker launcher mounts the host cache into that user's container home.
For another location, add ``asset_root`` to ``shared.environment`` in the
Experiment Definition. Keep the source assets available for the full run.


Run and extend the evaluation
-----------------------------

Prepare the supported Arena Docker runtime using
:doc:`../quickstart/installation`. Run these commands as the host user inside
the checkout's container, from ``/workspaces/isaaclab_arena``.

The eight-scenario verification experiment uses zero actions and a two-second
budget per Run:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/return_to_service/experiment_configs/full.yaml

Each Run should record one completed, unsuccessful episode. This exercises
scene construction, timeout handling, metrics, and HTML reporting.
``smoke.yaml`` provides the same check for one scenario. To record the
workcell, add ``shared.environment.enable_cameras=true --record_camera_video``.

For a candidate policy, copy the Experiment Definition, replace
``shared.policy`` with its registered type or dotted class path and typed
configuration, and set ``shared.environment.episode_length_s=600.0``. Keep
that budget and the observation contract consistent across compared policies.
The environment's standalone default is also 600 seconds. Use
``shared.rollout_limit.num_episodes=5`` for repeated episodes, or
``runs.<name>.<path>=<value>`` to override a particular Run. See
:doc:`../concepts/concept_arena_experiments` for configuration precedence.

The runner writes ``arena_experiment_result.json`` and an HTML report with
per-Run metrics. Report success by scenario alongside simulated time,
unnecessary replacements, and isolation violations. A partial-completion
score does not establish a valid final kit.

The registered ``ReturnToServiceEnvironment`` factory composes an Arena scene,
the existing ``droid_differential_ik`` embodiment, and ``ReturnToServiceTask``.
The factory exposes typed asset-root, scenario, episode-budget, table-height,
and gripper-gain settings. The benchmark uses the embodiment's normal action
and camera interfaces. Its gripper driver uses 4 N m/rad stiffness and
1 N m s/rad damping; robot geometry, arm gains, mimic joints, limits, and the
existing disabled self-collision setting are preserved.


Arena integration
-----------------

``ReturnToServiceTask`` extends ``CompositeTaskBase``. Its six condition tasks
share Arena's progress tracker, current final-state checks, consecutive-step
predicate, per-subtask metrics, and episode timeout. The service-specific
conditions remain explicit because voltage certificates, isolation history,
and fault-dependent disposition cannot be represented by ordinary pick-and-place
success alone.

``ServiceRuntime`` implements Arena's optional ``TaskRuntime`` lifecycle.
Predicates and observations only read the resulting snapshot. Arena updates it
before success evaluation, releases its connectors before reset events, then
initializes the selected episodes after placement and variation events. The
runtime uses ``ArenaWorld``, shared collision geometry, and the full relative-pose
predicate, including angular alignment. Tray volumes and unused-part return
targets follow live fixture frames.

The shared ``PlaceInRegionTask`` is available for independent placement tasks
using these assets. It checks full supported collision shapes, settling, and
optional measured gripper release. The complete service benchmark additionally
checks unexpected case contents, keyed packing poses, component certificates,
and isolation history.

Variation inspection and replay
-------------------------------

Inspect effective paths before selecting factors:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/return_to_service/experiment_configs/full.yaml \
     --list_variations --variations_format json \
     --variations_output /tmp/service_variations.json

This inspection starts SimulationApp and constructs the environment definition.
Read the output file directly; simulator diagnostics remain in the console.
The structured catalogue includes disabled variations and explains task-specific
restrictions. Disappearance is rejected for required inventory. Existing fixed
scenario Runs preserve their work order and conditions; randomized conditions
are evaluator configuration and are not added to policy observations.

The companion ``experiment_configs/variations.yaml`` separates each factor into
one named Run. These are short zero-action verification runs; configure a policy
and a 600-second episode budget for evaluation.

.. list-table:: Variation examples
   :header-rows: 1
   :widths: 25 75

   * - Run
     - Configuration
   * - ``sampled_conditions``
     - Samples the eight fault scenarios independently at episode reset.
   * - ``battery_mass``
     - Uses Arena's mass variation for original and spare batteries, within 10 percent of nominal mass.
   * - ``camera_translation``
     - Uses Arena's wrist-camera extrinsics variation within 2 mm on each axis.
   * - ``translated_left``
     - Moves the workstation group 10 mm along positive Y at build time.
   * - ``rotated_right``
     - Rotates that group by minus one degree about the cradle neighborhood at build time.
   * - ``lighting``
     - Uses Arena's build-time intensity and color variations on the authored studio dome.

Each layout moves fixtures and contents coherently, retaining their local socket
and region frames. The robot, bench, floor, and lighting stay fixed. Arena's
``PlacementLayouts`` restores the selected layout's writable physics roots on
reset; static fixtures select their pose at build time. The narrow layout bank
preserves the compact workcell's reach constraints. It does not establish
collision-free trajectories between interaction poses.

Set ``environment_builder.variation_seed`` to reproduce enabled built-in sampler
draws independently of unrelated factors. Replay a Run's
``variation_samples_rebuild0.jsonl`` using ``variation_replay_path`` with the same
environment IDs and episode schedule. The companion includes the final automatic
reset draw. It does not replay robot motion, contacts, or arbitrary physical
state. Placement artifacts and variation artifacts serve different purposes.


Measurements and scoring
------------------------

PhysX simulates rigid parts, contact, spring-return buttons, and the case's
passive hinge and latch. Component detents engage physical constraints after
measured alignment and contact checks. The case catch engages when measured
lid and latch angles are closed. The runtime does
not place components by writing their poses during an episode; reset restores
initial physical state. Passive guides support the filter and removable
adapter, and the obstruction rests within a collision-checked inlet.

Battery load voltage and airflow are functional models driven by measured
assembly state and component conditions. They do not simulate electrochemistry
or fluid dynamics. Thresholds are benchmark parameters, not commercial vacuum
specifications. Tests require a physical button press and a stable test
interval; changed or incomplete assemblies invalidate an in-progress test.

The public ``instruments.readings`` observation exposes battery voltage,
battery result code, airflow in liters per second, and airflow result code.
Codes are ``0`` idle, ``1`` running, ``2`` pass, ``3`` fail, and ``4`` invalid;
unavailable measurements are ``-1``. Displays retain their last result, so a
displayed pass does not certify a replacement component. Keep scenario names,
hidden fault flags, object identities, and evaluator diagnostics outside a
candidate policy's observation adapter. Report whether the policy uses numeric
instrument observations or reads instruments only through images.

Success requires the complete current state to remain valid for 15 consecutive
control steps: valid tests and certificates, preserved healthy originals,
complete packing, cleanup, battery-isolation history, and released objects at
rest. Historical milestones alone cannot complete an episode.

Containment checks use each configured physical box or cylinder in the current
case or tray frame. The case floor allows 50 micrometers of resting contact
penetration; its other faces retain a one-micrometer numerical tolerance.
Pose, velocity, release, and unexpected-object intrusion checks apply
independently. These allowances accommodate simulation contacts without
accepting an object that protrudes through a case wall.

``service_diagnostics`` records the following columns:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Metric
     - Meaning
   * - ``completion_fraction``
     - Fraction of six current terminal conditions satisfied; not task success.
   * - ``battery_tests``
     - Load-test attempts, including invalid or interrupted attempts.
   * - ``airflow_tests``
     - Airflow-test attempts, including invalid or interrupted attempts.
   * - ``unnecessary_replacements``
     - Non-original insertions when the corresponding original was healthy.
   * - ``elapsed_seconds``
     - Simulated episode time.
   * - ``isolation_violation``
     - Whether the airway was accessed with the battery connected.

A diagnostic substitution can be recovered by restoring the healthy original
and retesting, but its replacement count remains recorded. Metrics are
averaged over recorded episodes; retain the scenario breakdown when comparing
policies.


Verify and adapt the benchmark
------------------------------

Run the simulator-independent checks in a separate Python process:

.. code-block:: bash

   /isaac-sim/python.sh -m pytest -q \
     isaaclab_arena/tests/test_return_to_service_model.py \
     isaaclab_arena/tests/test_return_to_service_measurements.py \
     isaaclab_arena/tests/test_return_to_service_sampling.py \
     isaaclab_arena/tests/test_return_to_service_connectors.py \
     isaaclab_arena/tests/test_return_to_service_containment.py \
     isaaclab_arena/tests/test_return_to_service_case_containment.py \
     isaaclab_arena/tests/test_return_to_service_collision_geometry.py \
     isaaclab_arena/tests/test_return_to_service_asset_contract.py \
     isaaclab_arena/tests/test_return_to_service_asset_fits.py \
     isaaclab_arena/tests/test_return_to_service_asset_uvs.py

After generating the assets, run the physical checks:

.. code-block:: bash

   /isaac-sim/python.sh -m pytest -sv \
     isaaclab_arena/tests/test_return_to_service_runtime.py \
     isaaclab_arena/tests/test_return_to_service_metrics.py \
     isaaclab_arena/tests/test_return_to_service_evaluator_integration.py

The tests cover faults and certificate invalidation, adversarial geometry,
measured retention and release, reset isolation, and public observations. The
integration test explicitly positions physical fixtures and actuates buttons
to verify instrument dwell, all eight successful service states, interrupted
success holds, terminal HDF5 metrics, and autoresets. This isolates evaluator
correctness from robot manipulation; it does not certify collision-free
trajectories between the fixture states. The zero-action experiment separately
checks unsuccessful episode reporting.

The implementation separates responsibilities:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Modules
     - Responsibility
   * - ``asset_source/``
     - Blender geometry, materials, affordances, export, and validation.
   * - ``assets.py``, ``scene.py``
     - Cached physics layers and shared workcell layout.
   * - ``connectors.py``
     - Measured capture, physical retention, and release.
   * - ``isaaclab_arena.geometry``
     - Shared batched physical geometry, parent-relative containment, and overlap.
   * - ``model.py``, ``scenarios.py``
     - Simulator-independent instruments, certificates, faults, and success rules.
   * - ``runtime.py``
     - Physical measurements and one model update per control step.
   * - ``task.py``, ``metrics.py``
     - Public instruments, composite task criteria, and terminal metrics.

To add a fault, define its observable consequences in the model and measure
its physical evidence in the runtime. Keep the core model independent of a
particular policy. Test certificate and interlock rules separately from
physical contact, reset, and manipulation behavior.


Reuse the service assets
------------------------

The scene uses registered Arena asset adapters. They reference the existing
Blender bundle and defer all manifest and USD preparation until construction:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Registry name
     - Existing Arena interface
   * - ``return_to_service_bench``
     - Authored work surface exposed as a graph background.
   * - ``return_to_service_component``
     - Movable rigid parts; ``component`` selects a declared source asset.
   * - ``return_to_service_fixture``
     - Fixed bins, trays, panels, and work surfaces.
   * - ``return_to_service_instrument``
     - Kinematic cradle and test instruments, with their authored socket geometry.
   * - ``return_to_service_button``
     - ``Pressable`` over the physical spring-return ``press`` joint.
   * - ``return_to_service_case``
     - ``Openable`` over a selected ``hinge`` or ``latch`` joint.
   * - ``return_to_service_lighting``
     - Authored studio dome with existing intensity and RGB variations.

Inspect their typed parameters, affordances, and reset semantics using the
:doc:`authoring discovery interfaces <../concepts/agentic_environment_generation/authoring_discovery>`:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode catalog --format json
   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode validate --format json \
      --env_spec isaaclab_arena_environments/return_to_service/authoring_examples/battery_in_bin.yaml

The small graph example uses the existing DROID embodiment and the Blender-authored
service bench, lighting, battery, and bin with ``PlaceInRegionTask``.
It requires collision-shape containment, settling, and
measured gripper release. Its explicit region bounds match the bin manifest's
interior; update those bounds if authoring a different bin. This is a reusable
placement example, separate from the full benchmark's diagnosis, retention,
certificate, and interlock requirements. A static validation pass does not
establish that a policy can complete it.

After preparing the asset bundle, the normal graph runner can build the example:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
      --mode build --visualizer none --num_envs 1 \
      --env_spec isaaclab_arena_environments/return_to_service/authoring_examples/battery_in_bin.yaml

For Python composites, ``case.for_joint("hinge")`` and
``case.for_joint("latch")`` produce task-only ``Openable`` views of the same
case. Add the original case once to the scene. The views share its scene key
and asset config; they do not spawn duplicate articulations. Arena's existing
joint utilities preserve the authored negative-opening lid and negative-travel
button conventions.

``lighting.intensity`` and ``lighting.color`` use the existing build-time
variation classes. They start disabled, preserving the authored baseline.
Enabling them creates a cached USD opinion over ``StudioDome``; the Blender
source file remains unchanged. Use ``--list_variations --variations_format
json`` on the benchmark runner to inspect effective values and restrictions.


Real-world basis
----------------

The `Bosch GAS 18V-1 product description
<https://www.bosch-professional.com/gb/en/products/gas-18v-1-06019C6200>`_
documents quick-release dust emptying. Its `maintenance instructions
<https://www.bosch-professional.com/manuals/professional/gb/en/online-manual/200658989/en-GB/1704931959618008843.html>`_
require battery removal before maintenance and list the collection cup,
filters, and battery capacity among the checks for insufficient suction.

`Kärcher's official outlet <https://www.karcheroutlet.co.uk/About-Us.asp>`_
sells refurbished vacuums. Its `refurbishment FAQ
<https://www.karcheroutlet.co.uk/FAQ.asp>`_ describes returned products being
tested by engineers and repackaged for sale. These sources ground the work
order. The Arena appliance and fixtures are original, simplified assets;
the example does not reproduce a manufacturer's product or claim an existing
autonomous refurbishment deployment.
