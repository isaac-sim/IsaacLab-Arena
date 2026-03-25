Docker Containers
=================

This first version of Isaac Lab Arena is designed to run inside a Docker container.


Isaac Lab Arena runs inside the **Base** container, which contains all Arena code and dependencies.
Tagged as ``isaaclab_arena:latest``.

To start the container run:

:docker_run_default:

The container will build (if needed) and drop you into an interactive shell.

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

For running the simulation and policy model in separate containers, see
:doc:`../advanced/groot_server_client`.
