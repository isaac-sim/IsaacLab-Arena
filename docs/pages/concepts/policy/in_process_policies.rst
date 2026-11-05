In-process policies
===================

In-process policies compute actions inside the simulation process. They need no server and no
network, so they are the fastest way to check an environment, replay data, or run a small trained
network.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Class
     - ``name``
     - Use it to
   * - ``ZeroActionPolicy``
     - ``zero_action``
     - Smoke-test an environment: zero actions, no model.
   * - ``ReplayActionPolicy``
     - ``replay``
     - Replay one episode from an HDF5 recording. It reports the episode length, so the runner
       stops when the recording ends.
   * - ``RslRlActionPolicy``
     - ``rsl_rl``
     - Run a trained RSL-RL checkpoint. Its ``params/agent.yaml`` is loaded alongside.

All three live in
`isaaclab_arena/policy/ <https://github.com/isaac-sim/IsaacLab-Arena/tree/main/isaaclab_arena/policy>`__.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 100 <env>
   python isaaclab_arena/evaluation/policy_runner.py --policy_type replay --replay_file_path demo.hdf5 <env>

Extend
------

Add an in-process policy when inference is cheap and its dependencies fit in the Arena image,
for example a scripted controller, an imitation-learning checkpoint, or another RL library.
Follow :ref:`writing-a-custom-policy`. Two hooks matter for this family:

* ``has_length()`` / ``length()``: return a fixed length for dataset-driven policies.
* ``reset(env_ids)``: clear recurrent or chunk state for the environments that reset.

If the model needs its own dependencies, GPU, or machine, use a
:doc:`server–client policy <server_client_policies>` instead.
