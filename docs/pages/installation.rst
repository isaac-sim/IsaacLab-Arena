Installation
============

At version ``v1.0.0`` ``isaac_arena`` only support installation from source inside a docker container.

As a developer, you need to launch the container and then run isaac_arena inside.

We have two versions of our container:

- **Without GR00T:** This container has the minimal set of dependencies for running ``isaac_arena``
  without GR00T.
- **With GR00T:** This container contains additional dependencies for running ``isaac_arena``
  with the GR00T policy.

To launch out container, run:

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
