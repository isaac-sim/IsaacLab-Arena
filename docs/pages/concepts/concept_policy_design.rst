Policy Design
=============

Policies define how agents generate actions from observations.
The policy system provides a registry system for managing different policies.

Core Architecture
-----------------

Policies use the ``PolicyBase`` abstract class:

.. code-block:: python

   class PolicyBase(ABC):
       @abstractmethod
       def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
           """Compute an action given the environment and observation.

           Args:
               env: The environment instance
               observation: Observation dictionary from the environment

           Returns:
               torch.Tensor: The action to take
           """

This enables seamless swapping between different policy implementations while maintaining consistent integration with IsaacLab Arena environments.

Policies in Detail
------------------

**Policy Categories**
   Three main types addressing different use cases:

   - **Zero Action Policy**: Baseline that returns zero actions for environment testing and physics validation
   - **Replay Action Policy**: Replays pre-recorded demonstrations from HDF5 datasets for analysis and evaluation
   - **GR00T Neural Policies**: Advanced foundation models for visuomotor control with multi-modal inputs

**Implementation Patterns**
   Common policy implementation approaches:

   - **Stateless Policies**: Pure functions from observations to actions (ZeroActionPolicy)
   - **Dataset-Driven**: Load and replay recorded trajectories (ReplayActionPolicy)
   - **Neural Networks**: Process visual and proprioceptive inputs for learned behaviors (GR00T policies)

Environment Integration
-----------------------

.. code-block:: python

   # Policy creation from CLI arguments
   arena_builder = get_arena_builder_from_cli(args)
   env = arena_builder.make_registered()
   policy, num_steps = create_policy(args)

   # Standard execution loop
   obs, _ = env.reset()
   for step in range(num_steps):
       with torch.inference_mode():
           actions = policy.get_action(env, obs)
           obs, rewards, terminated, truncated, info = env.step(actions)
           if terminated.any() or truncated.any():
               obs, _ = env.reset()

Usage Examples
--------------

**Baseline Testing**

.. code-block:: bash

   # Zero action policy for environment validation
   python policy_runner.py --policy_type zero_action kitchen_pick_and_place --num_steps 1000

**Demonstration Replay**

.. code-block:: bash

   # Replay recorded demonstrations
   python policy_runner.py --policy_type replay --replay_file_path demos.h5 kitchen_pick_and_place

**Neural Policy Execution**

.. code-block:: bash

   # GR00T foundation model deployment
   python policy_runner.py --policy_type gr00t_closedloop --policy_config_yaml_path config.yaml

**Custom Policy Integration**

.. code-block:: python

   class CustomPolicy(PolicyBase):
       def get_action(self, env, observation):
           # Custom control logic
           return torch.zeros(env.action_space.shape)

   policy = CustomPolicy()
   actions = policy.get_action(environment, observations)
