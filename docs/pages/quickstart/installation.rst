Installation
============

Isaac Lab Arena supports installation from source inside a Docker container.

You'll launch the container and run isaac_arena inside it.


1. Clone the repository and initialize submodules:

:isaaclab_arena_git_clone_code_block:

.. code-block:: bash

    git submodule update --init --recursive

3. Launch the docker container:

.. code-block:: bash

    ./docker/run_docker.sh


4. Optionally verify installation by running tests:

.. code-block:: bash

    pytest -s isaac_arena/tests/

You're ready to run your first IsaacLab-Arena example!
