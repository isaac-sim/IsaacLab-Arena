GR00T
=====

`GR00T N1.6 <https://github.com/NVIDIA/Isaac-GR00T/>`_ is a pre-trained robotic
foundation model. No fine-tuning or separate model download is required. The weights
are fetched from `HuggingFace <https://huggingface.co/nvidia/GR00T-N1.6-DROID>`_
when the policy server starts for the first time.

Start a GR00T policy server
---------------------------

The closed-loop policy connects to a GR00T policy server in a separate process. The
server runs from the
`Isaac-GR00T <https://github.com/NVIDIA/Isaac-GR00T/tree/e29d8fc50b0e4745120ae3fb72447986fe638aa6>`_
submodule pinned at commit ``e29d8fc``. Populate it if needed:

.. code-block:: bash

   git submodule update --init submodules/Isaac-GR00T

.. note::

   Blackwell GPUs with compute capability ``sm_120`` require CUDA 12.8 or newer. The
   `official GR00T documentation <https://github.com/NVIDIA/Isaac-GR00T/blob/e29d8fc50b0e4745120ae3fb72447986fe638aa6/README.md?plain=1#L102>`_
   specify CUDA 12.8 and ``pytorch-cu128`` for RTX 5090 systems. Please refer to the
   documentation for the latest requirements.

Then start the server from the repository root in a separate shell:

.. code-block:: bash

   cd submodules/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.6-DROID \
     --embodiment-tag OXE_DROID \
     --device cuda --host 127.0.0.1 --port 5555

GR00T N1.6-DROID provides its own modality configuration, so the command does not need
``--modality-config-path``. The first launch downloads the model weights; later launches
reuse the local cache. Leave this server running.


Run GR00T with the Experiment Runner
------------------------------------

Arena includes a one-Run YAML configuration for the first rollout. It selects the
DROID environment, connects the GR00T policy to the server, and stops after three episodes.

.. dropdown:: Configuration file (``droid_pnp_gr00t_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml
      :language: yaml

GR00T N1.6-DROID uses absolute joint positions. The YAML therefore selects
``droid_abs_joint_pos`` and enables the cameras required by the policy. The natural-language
instruction belongs to the environment builder, while the server connection belongs to the
policy.

Open another shell and prepare the Arena runtime from the repository root, using either a native
``uv`` environment or the base Docker container (see :doc:`../installation` for the full setup):

.. tab-set::

   .. tab-item:: Native uv
      :selected:

      The GR00T client is not part of a default sync, so select its dependency group:

      .. code-block:: bash

         uv sync --extra dev --group gr00t-client
         source .venv/bin/activate
         export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

   .. tab-item:: Docker Container

      :docker_run_default:

Then start the rollout with the Experiment Runner:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml

The Kit window shows the DROID arm acting on GR00T commands. After every episode, Arena
reports whether the pick-and-place task succeeded.

If the server runs on another host or port, override the declared policy value. For example:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_gr00t_experiment.yaml \
     runs.droid_pnp_gr00t.policy.remote_port=5556

Run GR00T N1.7-DROID on GB300
---------------------------

N1.7-DROID uses a separate inference environment and checkout. The Arena submodule
and the N1.6 examples above retain their existing versions. Merely changing the
N1.6 server's model path does not load N1.7.

From the Arena repository root in your inference environment, with Python 3.12
and ``uv`` available, prepare the CUDA 13 runtime:

.. code-block:: bash

   bash tools/gb300/prepare_gr00t_n1d7.sh

This installs PyTorch 2.9 with CUDA 13 and Transformers 4.57.3 into
``~/.cache/arena-gr00t-n17-runtime`` and checks out Isaac-GR00T commit
``51d4c89f72fda44cbf77285c6a8114b52676b8a1`` in
``~/.cache/arena-gr00t-n17-src``. Set ``GR00T_N1D7_RUNTIME`` and
``GR00T_N1D7_ROOT`` to choose other locations. It uses PyTorch SDPA without
FlashAttention, TensorRT, or compilation. This is an inference-only environment;
video dataset decoding and training dependencies are not installed.

The backbone, `Cosmos-Reason2-2B <https://huggingface.co/nvidia/Cosmos-Reason2-2B>`_,
requires Hugging Face access. Accept access on its model page and authenticate
in the inference environment before launching:

.. code-block:: bash

   ~/.cache/arena-gr00t-n17-runtime/bin/hf auth login

Start the policy server on a free port. On the split-GPU machine, set
``CUDA_VISIBLE_DEVICES`` to the GB300's UUID from ``nvidia-smi -L``:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=<inference-gpu-uuid> \
     ~/.cache/arena-gr00t-n17-runtime/bin/python tools/gb300/serve_gr00t_n1d7.py \
       --host 127.0.0.1 --port 5557 --seed 42

The default checkpoint is ``nvidia/GR00T-N1.7-DROID`` at revision
``05e7cc97e40dbd33b0890c35cc0214fcb0547ab5``. For another compatible checkpoint,
pass ``--model-path`` and, for a Hugging Face repository, ``--model-revision``.
Use ``--gr00t-root`` when the source checkout is at a custom location.

After the server reports readiness, verify its protocol endpoint:

.. code-block:: bash

   python isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py \
     --host 127.0.0.1 --port 5557 \
     --timeout-sec 60 --poll-interval-sec 5 --request-timeout-ms 5000

In the Arena simulation environment, run three episodes of the RoboLab spring-clamp
task. On a split-GPU machine, set ``CUDA_VISIBLE_DEVICES`` to the RTX GPU's UUID
before starting the Experiment Runner:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=<simulation-gpu-uuid> \
     python isaaclab_arena/evaluation/experiment_runner.py \
       --viz none \
       --experiment_config isaaclab_arena_environments/experiment_configs/robolab_clamp_gr00t_n1d7_experiment.yaml

This uses ``isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml``
with its original 70-second task horizon and success criteria. Change the port with
``runs.clamp_in_right_bin.policy.remote_port=5558`` and the episode count with
``runs.clamp_in_right_bin.rollout_limit.num_episodes=1``.

N1.7 uses the embodiment tag ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT``.
Arena reads the checkpoint's modality configuration through the server, supplies
normalized gripper state and the ``panda_link8`` pose relative to ``panda_link0``,
and applies GR00T's DROID Euler/rotation-6D convention. It consumes eight actions
from each 40-action prediction before observing again. The upstream processor
decodes relative predictions to absolute joint positions, so the simulator uses
``droid_abs_joint_pos`` without adding the current joints a second time.


Evaluate several object variations
----------------------------------

The multi-Run YAML evaluates nine combinations of pick-up object, destination, HDR background,
and language instruction. Its ``shared`` mapping keeps the GR00T policy and rollout settings in
one place. Every Run lists only what changes.

.. dropdown:: Configuration file (``droid_pnp_srl_gr00t_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml
      :language: yaml

Start all nine Runs with one command:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml

The runner executes the nine Runs in YAML order. It keeps one SimulationApp open, but builds a
fresh environment for every Run.

.. figure:: ../../../images/gr00t_droid_3x3_grid.gif
   :width: 100%
   :alt: 3x3 grid of GR00T N1.6 DROID Runs across different objects, backgrounds, and destinations
   :align: center

   Nine closed-loop Runs of GR00T N1.6 on the DROID embodiment. Each cell changes the
   pick-up object, HDR background, and destination.

When all Runs finish, Arena prints a summary table followed by a metrics report:

.. dropdown:: Example Run summary and metrics
   :animate: fade-in

   Metric values can vary between evaluations. This is example output:

   .. code-block:: text

      +---------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
      |               Run Name                |  Status   |       Policy Type       | Num Envs | Num Steps | Num Episodes | Num Rebuilds |
      +---------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+
      |   droid_pnp_srl_gr00t_billiard_hall   | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |    droid_pnp_srl_gr00t_blue_block     | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      | droid_pnp_srl_gr00t_alphabet_soup_can | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |      droid_pnp_srl_gr00t_orange       | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |       droid_pnp_srl_gr00t_lemon       | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      | droid_pnp_srl_gr00t_tomato_sauce_can  | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |  droid_pnp_srl_gr00t_mustard_bottle   | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |     droid_pnp_srl_gr00t_sugar_box     | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      |        droid_pnp_srl_gr00t_mug        | completed | gr00t_remote_closedloop |    1     |   None    |      3       |      1       |
      +---------------------------------------+-----------+-------------------------+----------+-----------+--------------+--------------+

      ======================================================================
      METRICS SUMMARY
      ======================================================================

      droid_pnp_srl_gr00t_alphabet_soup_can:
        num_episodes                            3
        object_moved_rate                  0.0000
        success_rate                       0.0000

      droid_pnp_srl_gr00t_billiard_hall:
        num_episodes                            3
        object_moved_rate                  0.3333
        success_rate                       0.0000

      droid_pnp_srl_gr00t_blue_block:
        num_episodes                            3
        object_moved_rate                  0.0000
        success_rate                       0.0000

      droid_pnp_srl_gr00t_lemon:
        num_episodes                            3
        object_moved_rate                  1.0000
        success_rate                       0.6667

      ...
      ======================================================================

These results show that zero-shot deployment of robotic foundation models remains
challenging. Recent
`[robolab] <https://gitlab-master.nvidia.com/xuningy/robolab/-/blob/main/docs/analysis.md>`_
results compare GR00T with other vision-language-action models.


View rollouts as an HTML report
-------------------------------

The runner builds an HTML report for the complete evaluation. Add ``--record_camera_video`` to
record one video per camera and episode, then use
``--serve_evaluation_report`` to open the report through a local HTTP server:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_srl_gr00t_experiment.yaml \
     --output_base_dir ./output \
     --record_camera_video \
     --serve_evaluation_report

You can rebuild and serve a report later by pointing the standalone tool at the output
directory. It selects the most recent evaluation:

.. code-block:: bash

   python isaaclab_arena/visualization/report.py --video_dir ./output


Next steps
----------

To go beyond the pre-trained GR00T N1.6 foundation model, such as fine-tuning on your own
teleoperation data, see :doc:`/pages/example_workflows/imitation_learning/index` for the
complete imitation-learning workflows.
