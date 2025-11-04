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


:docker_run_default_code_block:


for more details see :doc:`docker_containers`.

3. **Optionally verify installation by running tests:**

.. code-block:: bash

    pytest -s isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/

With ``isaaclab_arena`` installed and the docker running, you're ready to build your first IsaacLab-Arena Environment. See :doc:`first_arena_env` to get started.


Omniverse Authentication (Use of Internal Assets)
-------------------------------------------------

To allow the project to access Nvidia Omniverse assets or services (e.g., stored scenes, USD files, or extensions on a Nucleus server),
you must authenticate using an Omniverse API token.

.. note::

    This is only needed for assets that are not yet hosted on the public Nucleus server. Once we go public, this will not be needed.
    TODO(cvolk, 2025-11-03): Remove this once we have public assets.

Isaac Sim and other Nvidia Omniverse applications use two environment variables ``OMNI_USER`` and ``OMNI_PASS`` to authenticate automatically.

To generate those, follow the instructions on the `Omniverse Authentication documentation <https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html>`_.

.. code-block:: bash

    export OMNI_USER='$omni-api-token'
    export OMNI_PASS=<your_generated_api_token>

Where:

* ``OMNI_USER`` should be set to the literal string ``$omni-api-token`` (including the leading dollar sign)

* ``OMNI_PASS`` should be the actual token value you generated.
