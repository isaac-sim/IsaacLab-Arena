PerfLab Benchmarks
==================

This guide tells a PerfLab operator how to set up and run these non-OSMO Arena benchmarks on Arena's
``main`` branch:

1. ``zero_action`` without cameras.
2. ``zero_action`` with the maintained production DROID cameras.
3. Pi0.5 with the maintained production DROID cameras and a separate OpenPI server.
4. The same apple in every environment.
5. Ten fruit assets distributed across environments.
6. Cosmos with the maintained production DROID cameras and a separate Cosmos server.
7. GR00T N1.6 with the maintained production DROID cameras and a separate GR00T server.
8. The VLM agent goal and chunk policies, each with one and with four environments, calling a
   hosted NVIDIA inference endpoint.

Each command starts one fresh Arena Experiment Runner process on one simulator GPU. Pi0.5,
Cosmos, and GR00T also need separately started policy servers on the same machine. The Pi0.5 and
Cosmos servers run in their own containers; the GR00T server runs on the host. The VLM agent
workloads need no server, but they need an internal API key and network access to the NVIDIA
internal inference endpoint.

The configurations for workloads 1 to 7 live in ``isaaclab_arena_environments/performance/``, laid
out like RoboLab's: ``experiment_configs/`` holds the Experiments, ``tasks/`` the fruit tasks, and
``scenes/`` the scene they share. The VLM agent Experiments live in
``isaaclab_arena_benchmarking/experiment_configs/``.


What the experiments show
-------------------------

All experiments use a DROID robot at the Maple table. The camera and policy workloads place a
Rubik's cube in a bowl. The object workloads place fruit in a bowl.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Experiment
     - What runs
     - What it tells us
   * - Camera-free baseline
     - ``zero_action``; cameras off
     - Scene construction, physics stepping, throughput, and the maximum environment count without
       camera or policy cost.
   * - Production-camera baseline
     - ``zero_action``; the maintained three-camera DROID rig is on
     - The additional cost of producing the normal camera observations. The rig renders at its
       declared 1280x720.
   * - Pi0.5
     - Pi0.5 remote policy; the maintained three-camera DROID rig is on
     - End-to-end production evaluation cost, including camera rendering and calls to the OpenPI
       server.
   * - Homogeneous objects
     - ``zero_action``; cameras off; one apple asset
     - Performance when every parallel environment uses the same object.
   * - Heterogeneous objects
     - ``zero_action``; cameras off; ten fruit assets
     - Performance when parallel environments use different object assets.
   * - Cosmos
     - Cosmos remote policy; the maintained three-camera DROID rig is on
     - End-to-end production evaluation cost, including camera rendering and calls to the Cosmos
       server.
   * - GR00T
     - GR00T N1.6 remote closed-loop policy; the maintained three-camera DROID rig is on
     - End-to-end production evaluation cost, including camera rendering and calls to the GR00T
       server.
   * - VLM goal policy
     - ``VLMGoalPolicy`` on the internal endpoint; DROID cameras on; 1 or 4 environments
     - How often the goal policy calls the VLM and how long the rollout waits on it, alone and with
       concurrent requests.
   * - VLM chunk policy
     - ``VLMChunkPolicy`` with ``chunk_size: 3`` on the internal endpoint; DROID cameras on; 1 or 4
       environments
     - The same for the chunk policy, which requests every three steps.

The first two experiments use the same ``droid_rel_joint_pos`` embodiment, so comparing them
isolates the cost of enabling the production camera rig. Pi0.5 requires the ``droid_abs_joint_pos``
embodiment. Cosmos and GR00T also use ``droid_abs_joint_pos``. Therefore, compare a policy run with the
production-camera baseline only as a complete production-path comparison, not as a measurement of
policy inference alone.

All three DROID cameras are rendered in the Pi0.5 experiment. The policy request uses the main
external-camera image and the wrist-camera image; the second external-camera image is rendered but
is not sent to OpenPI.

The homogeneous and heterogeneous experiments use the same task, zero-action policy, and
camera-free setup. They come from ``tasks/same_object.yaml`` and ``tasks/mixed_object.yaml``, which
share ``scenes/fruit_bowl_maple_table.yaml`` and differ only in the fruit list. The homogeneous
run uses one apple in every environment. The heterogeneous run assigns its ten fruit assets
repeatedly in the listed order. Compare these two runs to measure the cost of heterogeneous
parallel environments. Do not compare them with the Rubik's-cube baseline, which uses a different
scene definition, different task assets, and an explicit HDR. All ten fruit assets appear when
``NUM_ENVS`` is at least 10.

The zero-action policy keeps the robot stationary. A completed zero-action run proves that the
performance pipeline worked; it is not expected to solve the task.


Installation
------------

Follow the `Isaac Lab-Arena installation guide
<https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html>`_ and use the
Docker setup. Pi0.5, Cosmos, and GR00T also require the policy servers described below.

The Arena image installs PyTorch built for CUDA 13.0, so the host needs an NVIDIA driver from the
R580 branch or newer. Check the driver version with ``nvidia-smi`` on the host. With an older
driver, every run stops at startup with ``The NVIDIA driver on your system is too old``.


1. Prepare a clean Arena checkout
---------------------------------

Clone Arena's ``main`` branch on the host:

.. code-block:: bash

   git clone --branch main --recurse-submodules \
     https://github.com/isaac-sim/IsaacLab-Arena.git
   cd IsaacLab-Arena

To benchmark a release instead, replace ``main`` with the release tag.

Check that the checkout is clean and record the Arena and submodule revisions:

.. code-block:: bash

   git status --short
   git rev-parse HEAD
   git submodule status --recursive

``git status --short`` should print nothing. A line from ``git submodule status`` must not begin
with ``-``, ``+``, or ``U``.


2. Start the Arena container
----------------------------

Choose an absolute host directory that PerfLab will retain after the job. Start the container from
the repository root and mount that directory at ``$HOME/eval`` inside the container:

.. code-block:: bash

   PERFLAB_OUTPUT_HOST=/absolute/path/to/perflab-output
   mkdir -p "${PERFLAB_OUTPUT_HOST}"

   ./docker/run_docker.sh -e "${PERFLAB_OUTPUT_HOST}"

For the VLM agent workloads, export ``NV_API_KEY`` in this host shell before running the launcher.
The launcher forwards the key only when it creates the container; see
:ref:`perflab-vlm-agent-policies`.

The first invocation builds the Arena image, which also downloads the Isaac Sim base image. Do that
before timing any trial. The launcher enters the Arena container interactively; leave that terminal
open.

The launcher runs ``xhost`` only when ``DISPLAY`` is set. These commands do not open a simulator
window, so leave ``DISPLAY`` unset on a headless worker. If the launcher stops at ``xhost``, treat
that as a setup problem rather than a benchmark result.

Inside the Arena container, verify that the source checkout is mounted and the simulator GPU is
visible:

.. code-block:: bash

   cd /workspaces/isaaclab_arena

   ./submodules/IsaacLab/isaaclab.sh -p -c "import isaaclab_arena; print(isaaclab_arena.__file__)"
   nvidia-smi
   test -d "$HOME/eval" && test -w "$HOME/eval"

The import path must be below ``/workspaces/isaaclab_arena``, and the final command must exit with
status zero. The ``-e`` mount is applied only when the launcher creates the container; attaching to
an older container does not change its mounts.

Every Python command in this guide runs ``./submodules/IsaacLab/isaaclab.sh -p`` from
``/workspaces/isaaclab_arena``. This works in the launcher's shell and in shells attached later
with ``docker exec``, where ``isaaclab.sh`` is not on ``PATH``. ``isaaclab.sh`` prints a
deprecation warning on every call; it does not affect the run.

Create only the common output root. Do not create a trial's exact output directory before its
command runs:

.. code-block:: bash

   PERFLAB_OUTPUT_ROOT="$HOME/eval/perflab-workloads"
   mkdir -p "${PERFLAB_OUTPUT_ROOT}"


3. Understand the trial variables
---------------------------------

Set these variables inside the Arena container before each command:

.. code-block:: bash

   NUM_STEPS=300
   NUM_ENVS=1

- ``NUM_STEPS`` stays at ``300`` for every measured trial.
- ``NUM_ENVS`` selects one point from the experiment's sweep.

The exact output directory passed to Arena must be missing or empty.

Each command writes two timing files: ``arena_experiment_timings.json`` in the directory passed to
``--experiment_output_directory``, and ``arena_run_timings.json`` in the Run's subdirectory. For
example, the first run writes
``${PERFLAB_OUTPUT_ROOT}/camera-free/envs-${NUM_ENVS}/steps-${NUM_STEPS}/arena_experiment_timings.json``
and ``camera_free_baseline/arena_run_timings.json`` below that directory.

Each timing record names one timer and gives its ``count``, ``mean_ms``, ``total_ms``, ``min_ms``,
``max_ms``, ``p10_ms``, ``p50_ms``, and ``p90_ms``. The ``step`` timer covers one full control
step; ``step/policy_inference`` and ``step/env_step`` split it into the policy call and the
environment step. ``arena_run_timings.json`` is a list of these records.
``arena_experiment_timings.json`` lists them under ``runs``, with the Run name added to each
record, and sums them per timer under ``totals``. The timers cover the rollout only. Isaac Sim
startup and scene construction are part of the trial wall-clock but have no timer of their own.

PerfLab's harness should capture stdout and stderr outside the Arena output directory. Start the
trial wall-clock immediately before ``isaaclab.sh`` and stop it when that process exits. Container
startup, image downloads, asset downloads, and policy-server startup are preparation and must not
be included in this trial time.


4. Zero action without cameras
------------------------------

This is the simplest reference workload. It needs only the Arena container and one simulator GPU.
No policy server is involved.

Run this command inside the Arena container:

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/camera_free_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/camera-free/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}"

In plain English, this starts a fresh Isaac Sim process, creates ``NUM_ENVS`` copies of the task
with no cameras, takes ``NUM_STEPS`` stationary actions, writes one result directory, and exits.

For the measured sweep, use these environment counts in order:

.. code-block:: text

   1, 64, 256, 1024, 2048

Run ``4096`` only if ``2048`` completes successfully. Run each count once before increasing
``NUM_ENVS``.


5. Homogeneous objects
----------------------

Every environment uses the same apple asset. Cameras and policy inference are off.

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/same_object_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/same-object/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}"

Use ``1, 64, 256, 1024, 2048``, followed by ``4096`` only if ``2048`` succeeds.


6. Heterogeneous objects
------------------------

The environments use ten fruit assets in a repeating order. Cameras and policy inference are off.

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/mixed_object_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/mixed-object/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}"

Use the same sweep as the homogeneous run: ``1, 64, 256, 1024, 2048``, then ``4096`` if ``2048``
succeeds.


7. Zero action with production cameras
--------------------------------------

This uses the same task and zero-action policy, but enables the maintained three-camera DROID rig.
The Experiment Runner detects the camera requirement from the YAML, so do not add an
``--enable_cameras`` flag. Do not add video flags or camera-resolution overrides.

Run this command inside the Arena container:

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/production_camera_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/production-camera/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}"

This repeats the same task and stationary actions, but renders the maintained production camera
rig. The difference from Experiment 1 shows the camera cost.

For the measured sweep, use these environment counts in order:

.. code-block:: text

   1, 16, 64, 128, 256

If ``256`` completes successfully, continue by doubling to ``512``, then ``1024``, and so on while
the preceding point succeeds.


8. Prepare the Pi0.5 policy server
----------------------------------

Before running Pi0.5, start OpenPI in a second terminal on the host, from the repository root:

.. code-block:: bash

   ./isaaclab_arena_policies/isaaclab_arena_openpi/docker/run_openpi_server.sh -v pi05 -p 8000

The script gives the server container access to every GPU on the machine and has no option to
restrict it to one. Record the GPU memory the server uses, as shown by ``nvidia-smi`` once the
server is ready.

On the first invocation, the wrapper may build the OpenPI server image and download the
approximately 11 GB Pi0.5 checkpoint. Complete that work before starting the measured clock.

Wait until the server prints:

.. code-block:: text

   INFO:websockets.server:server listening on 0.0.0.0:8000

Leave this terminal and server running throughout the Pi0.5 sweep. Do not include server startup
in the Arena trial time. Keep the server's checkpoint, GPU, endpoint, and warm state unchanged
throughout the sweep.


9. Pi0.5 with production cameras
--------------------------------

Return to the Arena container terminal. The default setup uses the OpenPI server at
``127.0.0.1:8000``:

.. code-block:: bash

   POLICY_HOST=127.0.0.1
   POLICY_PORT=8000

Run this command inside the Arena container:

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/pi05_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/pi05/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}" \
     "shared.policy.remote_host=${POLICY_HOST}" \
     "shared.policy.remote_port=${POLICY_PORT}"

This renders the production camera observations, sends one request per environment to OpenPI when
a new action chunk is needed, applies the returned actions, writes the result, and exits. The
current client sends environment requests one at a time and reuses each Pi0.5 action chunk for 15
simulation steps. That behavior is part of this end-to-end measurement, and the 300-step run
exercises repeated requests.

For the measured sweep, use these environment counts in order:

.. code-block:: text

   1, 16, 64, 128, 256

If ``256`` completes successfully, continue by doubling while the preceding point succeeds. Finish
the Pi0.5 sweep before stopping the OpenPI server. Stop it with Ctrl-C in the same terminal that
started the wrapper so the wrapper can clean up correctly.


10. Prepare the Cosmos policy server
------------------------------------

Stop the Pi0.5 server first because both servers use port ``8000``. Then start Cosmos in the
second terminal on the host:

.. code-block:: bash

   ./isaaclab_arena_policies/isaaclab_arena_cosmos/docker/run_cosmos_server.sh -p 8000

Like the OpenPI script, this script gives the server container access to every GPU on the machine.
The local client still uses ``127.0.0.1:8000``.

If the Cosmos image is not already available, PerfLab must provide ``HF_TOKEN`` while the helper
builds the image and downloads the checkpoint. Finish that work before starting the measured
clock. Wait until the server reports that it is listening on port ``8000``, then leave it running
throughout the Cosmos sweep.


11. Cosmos with production cameras
----------------------------------

Set the Cosmos endpoint inside the Arena container:

.. code-block:: bash

   COSMOS_HOST=127.0.0.1
   COSMOS_PORT=8000

Then run:

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/cosmos_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/cosmos/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}" \
     "shared.policy.remote_host=${COSMOS_HOST}" \
     "shared.policy.remote_port=${COSMOS_PORT}"

Use ``1, 16, 64, 128, 256``, then continue doubling while the preceding point succeeds. Stop the
Cosmos server with Ctrl-C after the sweep finishes.


12. Prepare the GR00T policy server
-----------------------------------

GR00T uses port ``5555``, so it does not conflict with the Pi0.5 or Cosmos server. Unlike those
servers, it runs on the host from the pinned ``submodules/Isaac-GR00T`` checkout and needs ``uv``
on the host. Populate the submodule if needed, then start the server in the second terminal on the
host, from the repository root:

.. code-block:: bash

   git submodule update --init submodules/Isaac-GR00T
   cd submodules/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.6-DROID \
     --embodiment-tag OXE_DROID \
     --device cuda --host 127.0.0.1 --port 5555

On the first invocation, ``uv`` creates the GR00T Python environment and the server downloads the
``nvidia/GR00T-N1.6-DROID`` checkpoint. Complete that work before starting the measured clock. See
:doc:`../quickstart/running_a_real_policy/gr00t` for hardware requirements.

The server prints its configuration when it starts. Before the sweep, confirm that it answers
requests:

.. code-block:: bash

   uv run python ../../isaaclab_arena_policies/isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py \
     --host 127.0.0.1 --port 5555 \
     --timeout-sec 60 --poll-interval-sec 5 --request-timeout-ms 5000

Leave the server running throughout the GR00T sweep, and record the GPU memory it uses, as shown by
``nvidia-smi``.


13. GR00T with production cameras
---------------------------------

Set the GR00T endpoint inside the Arena container:

.. code-block:: bash

   GR00T_HOST=127.0.0.1
   GR00T_PORT=5555

Then run:

.. code-block:: bash

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/performance/experiment_configs/gr00t_benchmark_experiment.yaml \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/gr00t/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
     --viz none \
     --device cuda:0 \
     "shared.environment_builder.num_envs=${NUM_ENVS}" \
     "shared.rollout_limit.num_steps=${NUM_STEPS}" \
     "shared.policy.remote_host=${GR00T_HOST}" \
     "shared.policy.remote_port=${GR00T_PORT}"

The client reuses each GR00T action chunk for 32 simulation steps, so the 300-step run exercises
repeated requests.

Use ``1, 16, 64, 128, 256``, then continue doubling while the preceding point succeeds. Stop the
GR00T server with Ctrl-C after the sweep finishes.


.. _perflab-vlm-agent-policies:

14. VLM agent policies
----------------------

These four Experiments run the VLM agent goal and chunk policies on the Rubik's-cube task with the
``droid_abs_ik`` embodiment and the DROID cameras. :doc:`../quickstart/running_a_real_policy/vlm_agent`
explains how the policies work. There is no local policy server: every request goes to the NVIDIA
internal inference endpoint, ``https://inference-api.nvidia.com``, with the model
``openai/openai/gpt-6-astra``.

.. list-table::
   :header-rows: 1
   :widths: 45 15 20 20

   * - ``VLM_EXPERIMENT``
     - Environments
     - Rollout limit
     - Run name
   * - ``vlm_goal_policy_benchmark_experiment``
     - 1
     - 10 episodes
     - ``vlm_goal_pnp``
   * - ``vlm_goal_policy_parallel_envs_benchmark_experiment``
     - 4
     - 12 episodes
     - ``vlm_goal_pnp_parallel_envs``
   * - ``vlm_chunk_policy_benchmark_experiment``
     - 1
     - 10 episodes
     - ``vlm_chunk_pnp``
   * - ``vlm_chunk_policy_parallel_envs_benchmark_experiment``
     - 4
     - 12 episodes
     - ``vlm_chunk_pnp_parallel_envs``

These workloads differ from the others in three ways:

- **No sweep.** Each Experiment fixes its environment count. The single-environment runs time each
  request on its own. The four-environment runs stay below the policy's limit of eight concurrent
  requests, so when several environments need a response on the same step, all their requests are
  sent at once. Do not override ``num_envs``.
- **Episodes, not steps.** Each run stops after a fixed number of episodes, so ``NUM_STEPS`` does not
  apply and the number of steps differs between trials. An episode ends on success, at the task's
  70-second time limit (1,050 steps), or earlier if the cube falls off the table. A failed episode
  keeps requesting until it ends.
- **The endpoint dominates the time.** The simulation waits while a request is in flight, and each
  request takes several seconds. Most of the trial time is spent waiting on the hosted model, so
  the trial time depends on the endpoint's load as well as on the machine. The chunk policy makes
  many more requests than the goal policy and takes much longer. In one validation run, a goal
  episode made 10 requests and a chunk episode 133, at 10–16 seconds each. At that rate, the
  single-environment goal run takes about half an hour and the single-environment chunk run
  several hours.

Prepare the API key
~~~~~~~~~~~~~~~~~~~

From the NVIDIA network, generate an internal API key at
`inference.nvidia.com key management <https://inference.nvidia.com/key-management>`_. The requests
count against the token usage of the key's owner. Export the key on the host **before** starting
the Arena container in step 2:

.. code-block:: bash

   export NV_API_KEY="YOUR_INTERNAL_API_KEY"

``./docker/run_docker.sh`` passes ``NV_API_KEY`` into the container only when it creates the
container. Attaching to a running container does not add it. Inside the Arena container, check that
the key is set:

.. code-block:: bash

   test -n "${NV_API_KEY}" && echo "NV_API_KEY is set"

If it is not set, exit, stop the container, export the key on the host, and start the container
again.

Run the workloads
~~~~~~~~~~~~~~~~~

Set ``VLM_EXPERIMENT`` to one name from the table above, then run this command inside the Arena
container:

.. code-block:: bash

   VLM_EXPERIMENT=vlm_goal_policy_benchmark_experiment

   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       "isaaclab_arena_benchmarking/experiment_configs/${VLM_EXPERIMENT}.yaml" \
     --experiment_output_directory \
       "${PERFLAB_OUTPUT_ROOT}/vlm/${VLM_EXPERIMENT}" \
     --viz none \
     --device cuda:0

Run the four Experiments one after another, not at the same time, so their requests do not compete
for the endpoint. When the policy starts, it prints the endpoint and model and sends one short check
request. A missing or wrong key fails at that point, before the rollout starts.

Read the VLM timers
~~~~~~~~~~~~~~~~~~~

Besides the ``step`` timers, ``<run-name>/arena_run_timings.json`` holds three timers for the VLM:

- ``step/policy_inference/vlm_inference``: the steps that waited on the VLM. Its ``count`` is the
  number of such steps, and its ``total_ms`` is the rollout's total wait.
- ``step/policy_inference/vlm_inference/vlm_requests``: the rounds of concurrent requests. A round
  that returns an invalid command is followed by a correction round, so ``count`` can exceed the
  number of steps that waited.
- ``inference_request``: the individual requests, including retries. Its ``count`` is the number of
  requests, and its latency fields describe one request. It is recorded on the request threads, so
  it is not nested under ``step``.

The number of requests and tokens is not written to the output. Read token usage from the key's
Inference Hub usage page if needed.


15. Decide whether a trial passed
---------------------------------

A successful trial has all of the following:

- Process exit code ``0``.
- ``arena_experiment_result.json`` with ``runs.<run-name>.status`` equal to ``completed``.
- ``arena_experiment_timings.json``.
- ``<run-name>/arena_run_timings.json``.
- ``index.html``.
- ``<run-name>/episode_results_rebuild0.jsonl``. It may be empty when no episode finishes during
  the fixed number of steps.
- No CUDA OOM, host OOM, GPU reset, fatal renderer error, or policy-connection error in the log.

Expected Run names are:

.. list-table::
   :header-rows: 1

   * - Experiment
     - Run name
   * - Camera-free baseline
     - ``camera_free_baseline``
   * - Homogeneous objects
     - ``same_object_control``
   * - Heterogeneous objects
     - ``mixed_object_workload``
   * - Production-camera baseline
     - ``production_camera_baseline``
   * - Pi0.5
     - ``pi05_evaluation``
   * - Cosmos
     - ``cosmos_evaluation``
   * - GR00T
     - ``gr00t_evaluation``
   * - VLM agent policies
     - See the table in :ref:`perflab-vlm-agent-policies`.

In ``<run-name>/arena_run_timings.json``, the ``count`` of the ``step``, ``step/env_step``, and
``step/policy_inference`` records must equal ``NUM_STEPS``. The VLM agent runs stop after a number
of episodes instead, so these three counts must equal each other but have no fixed value.
