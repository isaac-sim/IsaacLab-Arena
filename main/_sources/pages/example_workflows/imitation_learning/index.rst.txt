Imitation Learning
==================

The following workflows demonstrate end-to-end imitation learning with Isaac Lab Arena,
covering teleoperation data collection, data generation, policy post-training, and
closed-loop evaluation.

Currently, the following imitation learning workflow examples are provided:

* :doc:`G1 Loco-Manipulation Box Pick and Place Task <../locomanipulation/index>`
* :doc:`G1 Static Apple-to-Plate Task <../static_apple/index>`
* :doc:`GR1 Open Microwave Door Task <../static_manipulation/index>`
* :doc:`GR1 Sequential Pick & Place and Close Door Task <../sequential_static_manipulation/index>`


GR00T Native Environment
------------------------

Arena simulation, data conversion, and evaluation clients use the standard **Base** container:

.. code-block:: bash

   ./docker/run_docker.sh

GR00T finetuning and GR00T policy servers use the native Isaac-GR00T ``uv`` environment
from ``submodules/Isaac-GR00T`` instead of Arena's Docker environment. Open another terminal
outside the Arena Base Docker container, ``cd`` to the Isaac-GR00T checkout, and set up the
environment by following the
`GR00T installation guide <https://github.com/NVIDIA/Isaac-GR00T#installation-guide>`_.


.. toctree::
   :maxdepth: 1

   ../locomanipulation/index
   ../static_apple/index
   ../static_manipulation/index
   ../sequential_static_manipulation/index
