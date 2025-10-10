Installation
============

At version ``v1.0.0`` ``Isaac Lab Arena`` only support installation from source inside a docker container.

As a developer, you need to launch the container and then run isaac_arena inside.

We have two versions of our container:

- **Without GR00T:** This container has the minimal set of dependencies for running ``isaac_arena``
  without GR00T.
- **With GR00T:** This container contains additional dependencies for running ``isaac_arena``
  with the GR00T policy.

To launch our container. First clone the repository:


:isaaclab_arena_git_clone_code_block:


and then launch the container:

.. tabs::
    .. tab:: Without GR00T

        .. code-block:: bash

            ./docker/run_docker.sh

    .. tab:: With GR00T

        .. code-block:: bash

            ./docker/run_docker.sh -g

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

   pytest -s isaac_arena/tests/

You're all set! You can now run some examples.
