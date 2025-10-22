Installation
============

Isaac Lab Arena currently only supports installation from source inside a Docker container.

You'll launch the container and run isaac_arena inside it.

We have two container versions:

- **Default:** Includes minimal dependencies for basic isaac_arena
- **With GR00T Dependencies:** Includes additional dependencies for GR00T policy support

First clone the repository:


:isaaclab_arena_git_clone_code_block:


Launch the Container
--------------------

.. tabs::
    .. tab:: Default

        .. code-block:: bash

            ./docker/run_docker.sh

    .. tab:: With GR00T Dependencies

        .. code-block:: bash

            ./docker/run_docker.sh -g

Optionally verify installation by running tests:

.. tabs::
    .. tab:: Default

        .. code-block:: bash

            pytest -s isaac_arena/tests/ --ignore=isaac_arena/tests/policy/

    .. tab:: With GR00T Dependencies

        .. code-block:: bash

            pytest -s isaac_arena/tests/

You're ready to run examples!
