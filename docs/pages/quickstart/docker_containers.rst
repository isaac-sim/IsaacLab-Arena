Docker Containers
=================

This first version of Isaac Lab Arena is designed to run inside a Docker container.


We provide two docker containers for Isaac Lab Arena:

- **Base**: Contains the Isaac Lab Arena code and all its dependencies.
- **Base + GR00T**: Additionally includes GR00T and its dependencies.

We include the two containers such that the user can choose between container with minimal
dependencies (**Base**) or container with all dependencies (**Base + GR00T**).

In order to start the containers run:

.. tabs::

    .. tab:: Base

        :docker_run_default_code_block:

    .. tab:: Base + GR00T

        :docker_run_gr00t_code_block:


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
