Policy
======

A policy in Arena is a standard interface between your model and the evaluation
pipeline. You implement one method — ``get_action(env, obs)`` — and the policy
plugs into the Experiment Runner without an ad-hoc inference loop for each
model. Arena's ``PolicyBase`` gives every typed policy the same execution
contract.

.. code-block:: python

   policy = ZeroActionPolicy(config=ZeroActionPolicyCfg())
   obs, _ = env.reset()
   action = policy.get_action(env, obs)

Built-in policies
-----------------

Arena ships with four policies:

**ZeroActionPolicy** (``"zero_action"``)
   Returns a zero-filled action tensor. Useful for validating an environment
   without a trained model.

**ReplayActionPolicy** (``"replay"``)
   Replays actions from a recorded episode stored in an HDF5 file.

**RslRlActionPolicy** (``"rsl_rl"``)
   Runs inference with a trained RSL-RL checkpoint. Loads the checkpoint and
   its accompanying ``params/agent.yaml`` automatically.


Writing a custom policy
-----------------------

Define a typed ``PolicyCfg``, subclass ``PolicyBase`` with that config, set a
``name``, register it with its config, and implement ``get_action``:

.. code-block:: python

   from dataclasses import dataclass

   import gymnasium as gym
   import torch
   from gymnasium.spaces.dict import Dict as GymSpacesDict

   from isaaclab_arena.assets.register import register_policy
   from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


   @dataclass
   class MyPolicyCfg(PolicyCfg):
       device: str = "cuda:0"


   @register_policy
   class MyPolicy(PolicyBase[MyPolicyCfg]):
       name = "my_policy"

       def __init__(self, config: MyPolicyCfg):
           super().__init__(config)

       def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
           # Your model inference here
           return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))

Construct the policy by passing its typed configuration directly:

.. code-block:: python

   policy_cfg = MyPolicyCfg(device="cuda:0")
   policy = MyPolicy(policy_cfg)

Typed registration lets an Arena Experiment select ``my_policy`` and compose
its YAML values directly into ``MyPolicyCfg``. See
:doc:`concept_evaluation_types` for the local and OSMO execution routes.

.. note::

   ``policy_runner.py`` and JSON job configurations are deprecated
   compatibility interfaces. New policies should expose typed configuration;
   do not add policy-specific ``argparse`` methods.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_evaluation_types
   concept_sensitivity_analysis
