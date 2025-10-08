Policy Design
=============

Policies in Isaac Arena define how agents generate actions from observations, providing a unified framework for integrating different control strategies, learning algorithms, and demonstration replay systems. The policy system supports everything from simple baselines to sophisticated neural network models, enabling seamless experimentation across different approaches.

Core Architecture
-----------------

The policy system is built around the ``PolicyBase`` abstract class, which defines a standard interface for all control policies:

.. code-block:: python

   class PolicyBase(ABC):
       def __init__(self):
           """Base class for policies."""

       @abstractmethod
       def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
           """Compute an action given the environment and observation.

           Args:
               env: The environment instance
               observation: Observation dictionary from the environment

           Returns:
               torch.Tensor: The action to take
           """

This abstraction enables seamless swapping between different policy implementations while maintaining consistent integration with Isaac Arena environments.

Policy Types and Implementations
--------------------------------

Isaac Arena supports three main categories of policies, each addressing different use cases:

**Zero Action Policy**
   A baseline policy that always returns zero actions, useful for testing environment mechanics, physics validation, and establishing performance baselines.

   .. code-block:: python

      # Always returns zeros for environment action space
      policy = ZeroActionPolicy()
      action = policy.get_action(env, observation)  # Returns torch.zeros(env.action_space.shape)

**Replay Action Policy**
   Replays pre-recorded actions from HDF5 datasets, enabling demonstration replay, trajectory analysis, and data-driven evaluation.

   .. code-block:: python

      # Load and replay specific episodes from demonstration data
      policy = ReplayActionPolicy(
          replay_file_path="path/to/demos.h5",
          episode_name="episode_001"
      )

**GR00T Neural Policies**
   Advanced neural network policies using NVIDIA's GR00T foundation models for visuomotor control:

   - **Closed-loop Inference**: Real-time policy execution with visual and proprioceptive feedback
   - **Action Chunking**: Generates multiple future actions per observation for temporal consistency
   - **Multi-modal Input**: Processes camera images, joint states, and language instructions
   - **Embodiment Adaptation**: Supports different robot configurations through joint remapping

   .. code-block:: python

      # GR00T policy with closed-loop visual control
      policy = Gr00tClosedloopPolicy(
          policy_config_yaml_path="config/gr00t_config.yaml",
          num_envs=1,
          device="cuda"
      )


Policy Integration System
-------------------------

Policies integrate with Isaac Arena environments through a standardized runner system that handles policy execution, environment interaction, and data management:

**Policy Runner Framework**
   The ``policy_runner.py`` system provides a unified interface for policy execution:

   .. code-block:: python

      # Standard policy execution loop
      arena_builder = get_arena_builder_from_cli(args)
      env = arena_builder.make_registered()
      policy, num_steps = create_policy(args)

      for step in range(num_steps):
          with torch.inference_mode():
              actions = policy.get_action(env, obs)
              obs, rewards, terminated, truncated, info = env.step(actions)
              if terminated.any() or truncated.any():
                  obs, _ = env.reset()

**Command-Line Interface**
    CLI support for policy configuration and execution:

   .. code-block:: bash

      # Zero action baseline
      python policy_runner.py --policy_type zero_action kitchen_pick_and_place --num_steps 1000

      # Demonstration replay
      python policy_runner.py --policy_type replay --replay_file_path demos.h5 kitchen_pick_and_place

      # GR00T neural policy
      python policy_runner.py --policy_type gr00t_closedloop --policy_config_yaml_path config.yaml


Practical Usage Patterns
------------------------

Common policy usage patterns in Isaac Arena:

**Development and Testing**
   Use zero action policies to validate environment mechanics and establish baselines before deploying learned policies.

**Demonstration Collection**
   Record human demonstrations using teleoperation devices, then replay using replay policies for analysis and learning.

**Neural Policy Deployment**
   Deploy pre-trained GR00T models for complex visuomotor tasks, leveraging foundation model capabilities for manipulation and locomotion.


**Example Integration**

.. code-block:: python

   # Create environment with specific embodiment
   environment = IsaacArenaEnvironment(
       name="kitchen_manipulation",
       embodiment=G1(control_mode="joint_space"),
       scene=KitchenScene(),
       task=PickAndPlaceTask()
   )

   # Load appropriate policy for the task
   policy = Gr00tClosedloopPolicy(
       policy_config_yaml_path="configs/g1_kitchen_manipulation.yaml",
       num_envs=environment.num_envs
   )

   # Execute policy in environment
   obs, _ = environment.reset()
   for _ in range(1000):
       actions = policy.get_action(environment, obs)
       obs, rewards, dones, info = environment.step(actions)

The policy system in Isaac Arena provides a flexible foundation for robotic control research, supporting everything from simple baselines to state-of-the-art foundation models while maintaining consistent interfaces and seamless integration with the broader Isaac Arena ecosystem.
