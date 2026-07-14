Policy
======

A policy in Arena is a standard interface between your model and the evaluation
pipeline. You implement one method — ``get_action(env, obs)`` — and the policy
plugs into both the single-job runner and the Experiment Runner without any
changes to either. In bare IsaacLab you would write an ad-hoc inference loop
for each model; Arena's ``PolicyBase`` gives a consistent contract that all
runners depend on.

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

The typed registration lets the single-job runner generate CLI flags from
``MyPolicyCfg`` and lets the Experiment Runner convert the current
``Job.policy_config_dict`` representation into that same type. See
:doc:`concept_evaluation_types` for details.

Config fields named ``device`` or ``num_envs`` reuse the corresponding shared
runner flags, so their defaults must match the runner defaults.

.. note::

   ``policy_runner.py`` remains an argparse frontend, but policies do not
   implement argparse methods. The runner generates their flags from the
   registered config and reconstructs it before creating the policy.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_evaluation_types
   concept_sensitivity_analysis
