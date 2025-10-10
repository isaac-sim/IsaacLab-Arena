Installation
============

Isaac Lab Arena currently only supports installation from source inside a Docker container.

You'll launch the container and run isaaclab_arena inside it.

We have two container versions:

- **Without GR00T:** Minimal dependencies for basic isaaclab_arena
- **With GR00T:** Additional dependencies for GR00T policy support

First clone the repository:


:isaaclab_arena_git_clone_code_block:


Then launch the container:

.. tabs::
    .. tab:: Without GR00T

        .. code-block:: bash

            ./docker/run_docker.sh

    .. tab:: With GR00T

        .. code-block:: bash

            ./docker/run_docker.sh -g

Optionally verify installation by running tests:

.. tabs::
    .. tab:: Without GR00T

        .. code-block:: bash

            pytest -s isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/

    .. tab:: With GR00T

        .. code-block:: bash

            pytest -s isaaclab_arena/tests/

You're ready to run examples!
