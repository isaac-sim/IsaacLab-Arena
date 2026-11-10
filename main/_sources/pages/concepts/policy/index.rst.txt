Policy
======

The quickstart guides run the ``zero_action`` policy, which keeps the robot still, and then swap in
a trained model such as π0.5 or GR00T. That swap works because every policy in Arena sits behind
the same class, ``PolicyBase``. The policy runner and the Experiment Runner only call that
class, so they evaluate a recorded episode, a trained network, a model behind a server, a
foundation-model agent, or a generated program in exactly the same way.

.. code-block:: python

   policy = ZeroActionPolicy(ZeroActionPolicyCfg())
   obs, _ = env.reset()
   action = policy.get_action(env, obs)


The policy contract
-------------------

A policy has one required method, ``get_action(env, obs)``, which returns a
``(num_envs, action_dim)`` action tensor for the current step. The runner resets the policy with
the environment, passes in the task's language instruction, and then alternates between asking
for an action and stepping the environment. When some environments finish an episode, the runner
resets the policy for just those environments:

.. code-block:: text

   env.reset() ─► policy.reset() ─► policy.set_task_description(instruction)
   loop:  action = policy.get_action(env, obs) ─► env.step(action)
          for environments that reset: policy.reset(env_ids)

The remaining methods are optional. ``close()`` releases clients, files, or threads.
``has_length()`` and ``length()`` let a recorded policy set the rollout length.
``default_image_size(cfg)`` reports the camera resolution that the policy's preprocessing
targets. See
`policy_base.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/policy/policy_base.py>`__
for the full class.


Four kinds of policy
--------------------

Every policy implements the same contract. They differ in where the decision is made, and each
kind has its own page:

* :doc:`In-process policies <in_process_policies>` compute actions inside the simulation
  process. Examples are ``zero_action``, ``replay``, and ``rsl_rl``.
* :doc:`Server–client policies <server_client_policies>` send observations to a model server
  and receive actions. Examples are Cosmos, π0.5, GR00T, and DreamZero.
* :doc:`Agent as policy <agent_policies>` asks a hosted foundation model to reason over images
  and state, and then runs the command it returns. Examples are ``vlm_goal`` and ``vlm_chunk``.
* :doc:`Code as policy <code_policies>` runs a program that a model writes against a skill
  API. It is coming soon with a new benchmark.


.. _writing-a-custom-policy:

Write your own policy
---------------------

To add a policy, define a typed configuration, subclass ``PolicyBase`` with it, give it a
``name``, and register it:

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

Registration makes the policy available to both runners. ``policy_runner.py --policy_type
my_policy`` generates command-line flags from ``MyPolicyCfg``. An Experiment Definition selects
the policy by its import path and fills the same configuration from YAML:

.. code-block:: yaml

   policy:
     type: my_package.my_policy.MyPolicy
     device: cuda:0

Configuration fields named ``device`` or ``num_envs`` reuse the runner's shared flags, so their
defaults must match the runner defaults.


Next steps
----------

Start with :doc:`in_process_policies` to check an environment without a model. To evaluate a
foundation model, continue to :doc:`server_client_policies`. See
:doc:`Arena Experiments <../concept_arena_experiments>` to run a policy across many tasks and
variations.

.. toctree::
   :maxdepth: 1
   :hidden:

   in_process_policies
   server_client_policies
   agent_policies
   code_policies
