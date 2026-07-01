Running a Real Policy
=====================

The zero-action experiments keep the robot still and success rates at zero. In this
section we will see actual pre-trained models in action. Arena ships clients for
several foundation-model policy servers. On the Rubik's-cube pick-and-place task used
below, openpi's pi05 generally lands non-zero success rates zero-shot, while GR00T
N1.6-DROID gets close to zero on most object variants; start with openpi for a more
interactive first run, and try GR00T or DreamZero for a contrasting baseline.

.. toctree::
   :maxdepth: 1

   openpi
   gr00t
   dreamzero
