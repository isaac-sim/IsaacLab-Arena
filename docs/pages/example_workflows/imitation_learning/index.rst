Imitation Learning
==================

The following workflows demonstrate end-to-end imitation learning with Isaac Lab Arena,
covering teleoperation data collection, data generation, policy post-training, and
closed-loop evaluation.

Currently, the following imitation learning workflow examples are provided:

* :doc:`G1 Loco-Manipulation Box Pick and Place Task <../locomanipulation/index>`
* :doc:`GR1 Open Microwave Door Task <../static_manipulation/index>`
* :doc:`GR1 Sequential Pick & Place and Close Door Task <../sequential_static_manipulation/index>`


GR00T Container
---------------

Some steps in these workflows (policy post-training and evaluation) require the **Base + GR00T**
container, which includes the `GR00T model <https://github.com/NVIDIA/Isaac-GR00T/>`_ dependencies
in addition to the standard Arena Base container. To launch it:

.. code-block:: bash

   ./docker/run_docker.sh -g

Not every step requires this container — the workflow pages will tell you when to use it.


.. toctree::
   :maxdepth: 1

   ../locomanipulation/index
   ../static_manipulation/index
   ../sequential_static_manipulation/index
