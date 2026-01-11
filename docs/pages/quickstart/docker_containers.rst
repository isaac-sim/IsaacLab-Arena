Docker Containers
=================

This first version of Isaac Lab Arena is designed to run inside a Docker container.


We provide two docker containers for Isaac Lab Arena:

- **Base**: Contains the Isaac Lab Arena code and all its dependencies. Tagged as ``isaaclab_arena:latest``.
- **Base + GR00T**: Additionally includes GR00T and its dependencies. Tagged as ``isaaclab_arena:cuda_gr00t_gn16``.

We include the two containers such that the user can choose between container with minimal
dependencies (**Base**) or container with all dependencies (**Base + GR00T**).

In order to start the containers run:

.. tabs::

    .. tab:: Base

        :docker_run_default:

    .. tab:: Base + GR00T

        :docker_run_gr00t:



The run docker will build the container and then enter in interactive mode.

.. note::
    The container with all dependencies (**Base + GR00T**) is significantly larger than the container with minimal dependencies (**Base**),
    so it is recommended to use the **Base** container for development and the **Base + GR00T** container for GR00T policy post-training and evaluation.
    If you are not sure which container to use, we recommend using the **Base** container.
    If you want to use the **Base + GR00T** container for development, currently it is not supported to run on Blackwell GPUs, and DGX Spark.

Mounted Directories
-------------------

The run docker script will mount the following directories on the host machine to the container:

- **Datasets**: from host: ``$HOME/datasets`` to container: ``/datasets``
- **Models**: from host: ``$HOME/models`` to container: ``/models``
- **Evaluation**: from host: ``$HOME/eval`` to container: ``/eval``

In our examples, we download input datasets and pre-trained models.
It is useful to download these to a folder mapped on the host machine to avoid re-downloading
between restarts of the container.
These directories are configurable through argument to the run docker script.

For a full list of arguments see the ``run_docker.sh`` script at
``isaac_arena/docker/run_docker.sh``.

Remote policies and GR00T (Optional)
-------------------------------------

GR00T policy evaluation can run in two modes:

.. tabs::

    .. tab:: Single container (Base + GR00T)

        Use the **Base + GR00T** container to run both the simulation
        environment and the GR00T policy in a single process:

        .. code-block:: bash

           bash docker/run_docker.sh --gr00t

        This is the simplest setup.  The policy runs locally inside the
        same container as the Isaac Lab Arena environment.

    .. tab:: Remote mode (two containers)

        Alternatively, the simulation and the policy model can run in
        separate containers connected via a lightweight RPC protocol.
        This is useful when the policy environment has different
        dependencies or needs to run on a different machine.

        - The **Base** Isaac Lab Arena container, started via
          ``docker/run_docker.sh``.  This container does not need to
          install GR00T when you run policies in remote mode.
        - A separate **GR00T policy server** container, started via
          ``docker/run_gr00t_server.sh``, which builds an image from
          ``docker/Dockerfile.gr00t_server`` and runs the remote policy
          server entrypoint.

        A typical workflow is:

        1. Start the Base container for simulation and evaluation:

           .. code-block:: bash

              bash docker/run_docker.sh

        2. In a second terminal, start the GR00T policy server container:

           .. code-block:: bash

              bash docker/run_gr00t_server.sh \
                --host 127.0.0.1  \
                --port 5555 \
                --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \
                --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml

        3. Inside the Base container, run the evaluation script with a
           client-side remote policy (see the static manipulation
           example workflow for full command lines).

        This setup cleanly separates the Isaac Lab Arena simulation
        environment from the GR00T policy server environment.

If you want to host other policy models as remote servers, you can
follow the same pattern: create a dedicated server Dockerfile and
launcher script (similar to ``docker/Dockerfile.gr00t_server`` and
``docker/run_gr00t_server.sh``), and point it to a custom
``ServerSidePolicy`` implementation as described in
:doc:`../concepts/concept_remote_policies_design`.
