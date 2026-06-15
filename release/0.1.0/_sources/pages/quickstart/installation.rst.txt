Installation
============

This page describes how to install Isaac Lab Arena from source inside a Docker container.

Supported Systems
-----------------

Isaac Lab Arena runs on Isaac Sim ``5.1.0`` and Isaac Lab ``2.3.0``.
The dependencies are installed automatically during the Docker build process.
Hardware requirements for Isaac Lab Arena are shared with Isaac Sim, and are detailed in
`Isaac Sim Requirements <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.

**GR00T fine-tuning:** Our example workflows include (optional) fine-tuning
of the `GR00T model <https://github.com/NVIDIA/Isaac-GR00T/>`_.
These workflows have additional requirements, notably that they do not support Blackwell GPUs, and
require large amounts of GPU memory.
Specific additional requirements for GR00T fine-tuning are detailed in the respective workflow pages.


Installation via Docker
-----------------------


Isaac Lab Arena supports installation from source inside a Docker container.
Future versions of Isaac Lab Arena, we will support a larger range of
installation options.


1. **Clone the repository and initialize submodules:**

:isaaclab_arena_git_clone_code_block:

.. code-block:: bash

    git submodule update --init --recursive

2. **Launch the docker container:**


:docker_run_default:


for more details see :doc:`docker_containers`.

3. **Optionally verify installation by running tests:**

.. code-block:: bash

    pytest -sv -m with_cameras isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/
    pytest -sv -m "not with_cameras" isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/

With ``isaaclab_arena`` installed and the docker running, you're ready to build your first IsaacLab-Arena Environment. See :doc:`first_arena_env` to get started.


Omniverse Authentication (Use of Internal Assets)
-------------------------------------------------

To allow the project to access Nvidia Omniverse assets or services (e.g., stored scenes, USD files, or extensions on a Nucleus server),
you must authenticate using an Omniverse API token.

.. todo::

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
