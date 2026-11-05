Policy
======

A policy maps an observation to an action. Arena puts every policy behind one class,
``PolicyBase``, so the policy runner and the Experiment Runner can evaluate any of them without
changes. A policy can be a recording, a trained network, a model behind a server, an agent, or a
generated program.

.. code-block:: python

   policy = ZeroActionPolicy(ZeroActionPolicyCfg())
   obs, _ = env.reset()
   action = policy.get_action(env, obs)

The contract
------------

``PolicyBase[Cfg]`` in
`policy_base.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/policy/policy_base.py>`__
is generic over a typed ``PolicyCfg`` dataclass. The runner calls:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Contract
   * - ``get_action(env, obs)``
     - **Required.** Return a ``(num_envs, action_dim)`` tensor for the current step.
   * - ``reset(env_ids=None)``
     - Clear per-environment state when those environments reset. ``None`` means all.
   * - ``set_task_description(text)``
     - Receive the task's language instruction before the first step.
   * - ``close()``
     - Release clients, files, or threads.
   * - ``has_length()`` / ``length()``
     - Report a fixed episode length (recorded policies), which overrides ``num_steps``.
   * - ``default_image_size(cfg)``
     - Report the camera size the policy's preprocessing targets, or ``None``.

The runner loop is the whole integration surface:

.. code-block:: text

   reset env ─► policy.reset() ─► policy.set_task_description(...)
   loop:  action = policy.get_action(env, obs) ─► env.step(action)
          for envs that reset: policy.reset(env_ids)

Policy families
---------------

All four families implement the same contract. They differ in where the decision is made.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Family
     - Where the decision is made
     - Examples
   * - :doc:`In-process <in_process_policies>`
     - In the simulation process
     - ``zero_action``, ``replay``, ``rsl_rl``
   * - :doc:`Server–client <server_client_policies>`
     - In a model server, over the network
     - Cosmos, π0.5, GR00T, DreamZero
   * - :doc:`Agent <agent_policies>`
     - In a hosted foundation model that reasons over images and state
     - ``vlm_goal``, ``vlm_chunk``
   * - :doc:`Code <code_policies>`
     - In a program written by a model, run against a skill API
     - Coming with the CaP benchmark

.. _writing-a-custom-policy:

Writing a custom policy
-----------------------

Define a ``PolicyCfg`` dataclass, subclass ``PolicyBase`` with it, set ``name``, and register it:

.. code-block:: python

   from dataclasses import dataclass

   import torch

   from isaaclab_arena.assets.register import register_policy
   from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


   @dataclass
   class MyPolicyCfg(PolicyCfg):
       device: str = "cuda:0"


   @register_policy
   class MyPolicy(PolicyBase[MyPolicyCfg]):
       name = "my_policy"

       def get_action(self, env, observation) -> torch.Tensor:
           return torch.zeros(env.action_space.shape, device=env.unwrapped.device)

Registration is what makes the policy usable everywhere:

* ``policy_runner.py --policy_type my_policy`` generates CLI flags from ``MyPolicyCfg``.
* An Experiment Definition selects it by import path and fills the config from YAML:

  .. code-block:: yaml

     policy:
       type: my_package.my_policy.MyPolicy
       device: cuda:0

Config fields named ``device`` or ``num_envs`` reuse the runner's shared flags, so their defaults
must match. See :doc:`Arena Experiments <../concept_arena_experiments>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   in_process_policies
   server_client_policies
   agent_policies
   code_policies
