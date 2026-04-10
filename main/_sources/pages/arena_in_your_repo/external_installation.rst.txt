Installing IsaacLab-Arena in Your Repository
============================================

The recommended way to consume IsaacLab-Arena from an external project is to include it as an
**unmodified git submodule** and extend it purely through its registration API — without editing
any file inside the Arena source tree.

This is the currently recommended integration pattern until IsaacLab-Arena is available as a
published ``pip`` package, at which point the submodule will be replaced by a simple
``pip install isaaclab_arena``. The environment and asset extension patterns below will
remain unchanged.


Repository Layout
-----------------

A typical external repository looks like this:

.. code-block:: text

   my_project/
   ├── submodules/
   │   └── IsaacLab-Arena/          ← unmodified Arena submodule
   ├── my_package/
   │   ├── pyproject.toml
   │   ├── isaaclab_arena_environments/
   │   │   ├── __init__.py
   │   │   └── my_environment.py    ← custom environment class
   ├── docker/
   │   └── Dockerfile
   └── .gitmodules

Add the submodule with:

.. code-block:: bash

   git submodule add git@github.com:isaac-sim/IsaacLab-Arena.git submodules/IsaacLab-Arena


Dockerfile
----------

Your base image must already have **Isaac Sim** and **Isaac Lab** installed (e.g.
``nvcr.io/nvidia/isaac-sim:6.0.0``).

Copy the submodule into the image and install Arena
before your own package run ``pip install -e``.

.. code-block:: dockerfile

   # Base image must have Isaac Sim
   # e.g. FROM nvcr.io/nvidia/isaac-sim:6.0.0

   # Image must have Isaac Lab installed
   # e.g. RUN /isaaclab/isaaclab.sh -i

   # Copy the submodule into the image and install Arena
   COPY submodules/IsaacLab-Arena /opt/arena
   RUN /isaac-sim/python.sh -m pip install -e /opt/arena

   # Install your package after Arena is in place
   COPY my_package /workspace/my_package
   RUN /isaac-sim/python.sh -m pip install -e /workspace/my_package

See Arena's own `Dockerfile
<https://github.com/isaac-sim/IsaacLab-Arena/blob/main/docker/Dockerfile.isaaclab_arena>`_
for a complete reference, including Isaac Lab installation and optional GR00T dependencies.

.. note::

   Note that above we have assumed that Arena is being installed inside a Docker container.
   Of course, if you have an non-Docker environment that satisfies the prerequisites, you can
   install Arena directly into that environment, without the need for Docker.

Next Steps
----------

Once Arena is installed, see :doc:`external_environments` for how to define
your own environments in your repository.
