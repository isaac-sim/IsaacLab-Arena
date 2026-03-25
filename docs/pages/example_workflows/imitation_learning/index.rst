Imitation Learning
==================

The following workflows demonstrate end-to-end imitation learning with Isaac Lab Arena,
covering teleoperation data collection, data generation, policy post-training, and
closed-loop evaluation.

GR00T Container
---------------

Some steps in these workflows (policy post-training and evaluation) require the **Base + GR00T**
container, which includes the `GR00T model <https://github.com/NVIDIA/Isaac-GR00T/>`_ dependencies
in addition to the standard Arena Base container. To launch it:

.. code-block:: bash

   ./docker/run_docker.sh -g

Not every step requires this container — the workflow pages will tell you when to use it.

.. note::
   The Base + GR00T container does not support Blackwell GPUs and requires large amounts of GPU
   memory.

.. toctree::
   :maxdepth: 1

   ../locomanipulation/index
   ../static_manipulation/index
   ../sequential_static_manipulation/index
