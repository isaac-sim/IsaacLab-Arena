Installation
============

Isaac Lab Arena supports installation from source inside a Docker container.

.. note:: In future versions of Isaac Lab Arena, we will support a larger range of
    installation options.


1. **Clone the repository and initialize submodules:**

:isaaclab_arena_git_clone_code_block:

.. code-block:: bash

    git submodule update --init --recursive

2. **Launch the docker container:**

.. code-block:: bash

    ./docker/run_docker.sh

for more details see :doc:`docker_containers`.

3. **Optionally verify installation by running tests:**

.. code-block:: bash

    pytest -s isaaclab_arena/tests/

With ``isaaclab_arena`` installed and the docker running, you're ready to build your first IsaacLab-Arena Environment. See :doc:`first_arena_env` to get started.
