Installation
============

This page describes how to install Isaac Lab Arena, either natively with ``uv``
or from source inside a Docker container.

.. list-table:: Supported workflows by installation method
   :header-rows: 1

   * - Installation method
     - Evaluation
     - Imitation learning
     - Reinforcement learning
   * - Docker
     - ✓
     - ✓
     - ✓
   * - uv (Isaac Lab from source)
     - ✓
     - ✓
     - ✓
   * - uv (Isaac Lab from wheel)
     - ✓
     - ✗
     - ✗

Supported Systems
-----------------

Isaac Lab Arena runs on Isaac Sim ``6.0.0`` and Isaac Lab ``3.0.0``.
The dependencies are installed automatically by either workflow below.
Hardware requirements for Isaac Lab Arena are shared with Isaac Sim, and are detailed in
`Isaac Sim Requirements <https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html>`_.


Native uv developer setup
-------------------------

Isaac Lab Arena can be installed natively with `uv <https://docs.astral.sh/uv/>`_;
the committed lockfile pins the complete environment. Two flavors are
available, differing only in where Isaac Lab comes from:

- **Source flavor (recommended):** Isaac Lab is installed editable from the
  ``submodules/IsaacLab`` checkout.
- **Wheel flavor:** Isaac Lab is installed from the published wheel, which
  does not include Isaac Lab's RL/IL scripts.

Both flavors follow the same workflow — clone, sync, activate, run; only the
``uv sync`` line differs.

Clone the repository:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/isaac-sim/IsaacLab-Arena.git
    cd IsaacLab-Arena

Sync the environment and activate it, picking the flavor that matches your
workflow:

.. tab-set::

   .. tab-item:: Source (recommended)
      :selected:

      .. code-block:: bash

          uv sync
          source .venv/bin/activate

   .. tab-item:: Wheel

      .. code-block:: bash

          uv sync --no-default-groups --group isaaclab-from-wheel
          source .venv/bin/activate

      .. note::
         The wheel flavor does not support the
         :doc:`imitation learning </pages/example_workflows/imitation_learning/index>`
         and
         :doc:`reinforcement learning </pages/example_workflows/reinforcement_learning_workflows/index>`
         workflows: the published Isaac Lab wheel does not include the scripts
         they rely on. Use the source flavor for those workflows.

``uv sync`` creates a Python virtual environment in ``.venv/`` (pinned by
``.python-version``), installs Isaac Lab Arena and Isaac Lab (editable from
``submodules/IsaacLab`` in the source flavor, or from the published wheel),
and pulls the matching Isaac Sim, PyTorch, and Newton wheels at the versions
pinned by the committed lockfile.

.. note::
   The two flavors are mutually exclusive within the single ``.venv``: syncing
   one replaces the other. In the wheel flavor, run ``python``/``pytest`` in
   the activated environment rather than through ``uv run`` — a bare
   ``uv run`` re-syncs the environment back to the source flavor.

Accept the Isaac Sim EULA so the first launch is non-interactive:

.. code-block:: bash

    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

Launch a short zero-action rollout as a visual validation that things are
running:

.. code-block:: bash

    python isaaclab_arena/evaluation/policy_runner.py \
      --policy_type zero_action --num_steps 20 cube_goal_pose

Optionally, watch the rollout in the GUI visualizer by adding ``--viz kit`` (and
a few more steps so there is time to see it):

.. code-block:: bash

    python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit --policy_type zero_action --num_steps 200 cube_goal_pose

Optionally verify the installation by running the test phases (the same phases
the Docker workflow runs below):

.. code-block:: bash

    pytest -sv -m "not with_cameras and not with_subprocess" isaaclab_arena/tests/
    pytest -sv -m "with_cameras and not with_subprocess" isaaclab_arena/tests/
    pytest -sv -m with_subprocess isaaclab_arena/tests/

With ``isaaclab_arena`` installed you're ready to build your first environment;
see :doc:`first_arena_env`.


Installation via Docker
-----------------------


Isaac Lab Arena supports installation from source inside a Docker container.
Future versions of Isaac Lab Arena, we will support a larger range of
installation options.


1. **Clone the repository and initialize submodules:**

:isaaclab_arena_git_clone_code_block:

.. code-block:: bash

    git submodule update --init --recursive

2. **Launch the docker container:**

:docker_run_default:

The container will build (if needed) and drop you into an interactive shell.

.. note::
   The run docker script mounts the following directories from the host machine if they exist:

   - **Datasets**: ``$HOME/datasets`` → ``/datasets``
   - **Models**: ``$HOME/models`` → ``/models``
   - **Evaluation**: ``$HOME/eval`` → ``/eval``

   When mounted a user avoids re-downloading datasets and models between container restarts,
   so our suggestion is to create these directories on the host machine before running the container.
   Note that the path of the mounted directories are configurable — see ``docker/run_docker.sh``
   for the full list of arguments.

3. **Optionally verify installation by running tests:**

.. code-block:: bash

    pytest -sv -m "with_cameras and not with_subprocess" isaaclab_arena/tests/
    pytest -sv -m "not with_cameras and not with_subprocess" isaaclab_arena/tests/
    pytest -sv -m with_subprocess isaaclab_arena/tests/

With ``isaaclab_arena`` installed and the docker running, you're ready to build your
first IsaacLab-Arena Environment. See :doc:`first_arena_env` to get started.
