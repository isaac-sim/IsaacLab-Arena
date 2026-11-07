In-Process Policies
===================

An in-process policy computes its actions inside the simulation process. It needs no server
and no network connection, so it is the fastest way to check that an environment loads, to
replay recorded data, or to run a small trained network.

Arena includes three in-process policies. ``zero_action`` returns zero actions and needs no
model weights, so the robot stays still while the environment loads and renders. ``replay``
plays back one episode from an HDF5 recording. It reports the episode length, so the rollout
ends when the recording does. ``rsl_rl`` runs a trained RSL-RL checkpoint and loads the
``params/agent.yaml`` saved with it.

All three live in
`isaaclab_arena/policy/ <https://github.com/isaac-sim/IsaacLab-Arena/tree/main/isaaclab_arena/policy>`__.
``ZeroActionPolicy`` is the smallest complete example of the policy contract:

.. dropdown:: ``zero_action_policy.py``
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena/policy/zero_action_policy.py
      :language: python
      :lines: 6-


Run an in-process policy
------------------------

Select the policy with ``--policy_type``. Each field of its configuration becomes a flag:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type replay \
     --replay_file_path demo.hdf5 \
     <environment> <environment arguments>

:doc:`../../quickstart/environment_variations` runs ``zero_action`` through the Experiment
Runner instead.


Add your own
------------

Write an in-process policy when inference is cheap and its dependencies fit in the Arena image.
Examples are a scripted controller, an imitation-learning checkpoint, or a policy from another
RL library. Follow :ref:`writing-a-custom-policy`. If the policy keeps state across steps, such
as a recurrent hidden state or a queued action chunk, clear it for the affected environments in
``reset(env_ids)``. If it plays back data of a fixed length, implement ``has_length()`` and
``length()``.

When a model brings its own dependencies or needs its own GPU or machine, keep it out of the
simulation process and run it as a server instead.


Next steps
----------

Continue to :doc:`server_client_policies` to evaluate a foundation model running in a separate
process.
