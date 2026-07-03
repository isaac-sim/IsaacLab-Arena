Policy
======

A policy in Arena is a standard interface between your model and the evaluation
pipeline. You implement one method — ``get_action(env, obs)`` — and the policy
plugs into both the single-job runner and the batch eval runner without any
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
       device: str = "cuda"


   @register_policy(cfg_type=MyPolicyCfg)
   class MyPolicy(PolicyBase[MyPolicyCfg]):
       name = "my_policy"

       def __init__(self, config: MyPolicyCfg):
           super().__init__(config)

       def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
           # Your model inference here
           return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))

The policy contract itself does not depend on argparse. The current single-job
CLI still expects concrete policies to provide the deprecated
``add_args_to_parser`` and ``from_args`` compatibility adapters:

.. code-block:: python

   class MyPolicy(PolicyBase[MyPolicyCfg]):
       ...

       # Deprecated compatibility adapter for the current argparse frontend.
       @staticmethod
       def add_args_to_parser(parser):
           # Add any CLI arguments your policy needs, then return the parser
           return parser

       @staticmethod
       def from_args(args):
           return MyPolicy(MyPolicyCfg(device=args.device))

Once registered, select the policy by name on the command line:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type my_policy \
     ...

For policies not registered by name, pass a dotted Python path instead
(e.g. ``--policy_type mypackage.mypolicy.MyPolicy``). The runner will
import and instantiate the class directly.

The typed registration associates the policy with its configuration for legacy
dictionary-based evaluation. Typed callers can construct ``MyPolicyCfg`` and
pass it to ``MyPolicy`` directly, without going through argparse. See
:doc:`concept_evaluation_types` for details.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_evaluation_types
   concept_sensitivity_analysis
