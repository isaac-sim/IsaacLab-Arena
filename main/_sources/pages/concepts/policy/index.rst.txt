Policy
======

A policy in Arena is a standard interface between your model and the evaluation
pipeline. You implement one method — ``get_action(env, obs)`` — and the policy
plugs into both the single-job runner and the batch eval runner without any
changes to either. In bare IsaacLab you would write an ad-hoc inference loop
for each model; Arena's ``PolicyBase`` gives a consistent contract that all
runners depend on.

.. code-block:: python

   policy = ZeroActionPolicy(config=ZeroActionPolicyArgs())
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

Subclass ``PolicyBase``, set a ``name``, decorate with ``@register_policy``,
and implement ``get_action``:

.. code-block:: python

   import gymnasium as gym
   import torch
   from gymnasium.spaces.dict import Dict as GymSpacesDict

   from isaaclab_arena.assets.register import register_policy
   from isaaclab_arena.policy.policy_base import PolicyBase


   @register_policy
   class MyPolicy(PolicyBase):
       name = "my_policy"

       def __init__(self, config):
           super().__init__(config)

       def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
           # Your model inference here
           return torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))

       @staticmethod
       def add_args_to_parser(parser):
           # Add any CLI arguments your policy needs, then return the parser
           return parser

       @staticmethod
       def from_args(args):
           return MyPolicy(config=None)

Once registered, select the policy by name on the command line:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type my_policy \
     ...

For policies not registered by name, pass a dotted Python path instead
(e.g. ``--policy_type mypackage.mypolicy.MyPolicy``). The runner will
import and instantiate the class directly.

To use a custom policy in the batch eval runner's JSON config, define a
``config_class`` dataclass on the policy and implement ``from_dict()``.
This lets the runner instantiate the policy from a plain dict without
going through argparse. See :doc:`concept_evaluation_types` for details.

More details
------------

.. toctree::
   :maxdepth: 1

   concept_evaluation_types
   concept_remote_policies_design
