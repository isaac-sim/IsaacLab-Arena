Policy Training
---------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Training Command
^^^^^^^^^^^^^^^^

Training uses Arena's RL Games entrypoint directly. The ``--agent_cfg_path`` argument
points to the task-specific RL Games PPO config, which defines the network architecture
and optimization hyperparameters used for this workflow.

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py \
     --headless \
     --num_envs 4096 \
     --max_iterations 1000 \
     --agent_cfg_path isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml \
     nist_assembled_gear_mesh_osc \
     --rl_training_mode

The environment-specific flag ``--rl_training_mode`` comes after the environment name.
During training, it disables success termination so the policy can continue collecting
experience near successful insertions.

Checkpoints are written to ``logs/rl_games/NistGearInsertionOscRlg/<timestamp>/``.
The agent configuration is saved alongside as ``params/agent.yaml``, which the
evaluation scripts use together with the RL Games checkpoint.

.. tip::

   Remove ``--headless`` if you want to watch training in the viewer.


Policy Configuration
^^^^^^^^^^^^^^^^^^^^

The RL Games config defines both the policy network and the PPO hyperparameters, including:

- a feed-forward MLP backbone
- an LSTM recurrent layer stack
- PPO learning rate, clipping, and rollout length
- the RL Games experiment name used for logs/checkpoints

The configuration file is:

.. code-block:: text

   isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml

Key settings in this file include:

- ``network.mlp.units`` for the actor-critic MLP sizes
- ``network.rnn`` for the recurrent policy/value architecture
- ``config.learning_rate`` for PPO optimization
- ``config.max_epochs`` for the training horizon
- ``config.name`` for the RL Games experiment name

In this workflow, the experiment name is ``NistGearInsertionOscRlg``.

Monitoring Training
^^^^^^^^^^^^^^^^^^^

During training, RL Games prints a summary to the console for each iteration,
including rollout statistics, losses, and timing information. Checkpoints are
saved periodically according to the settings in the YAML config.

You should see output indicating that training has started:

.. code-block:: text

   step: 0
   fps step: ...
   fps total: ...
   epoch: 1
   mean rewards: ...
   mean episode lengths: ...

If you are tuning the policy, the most useful signals to watch are:

- the reward terms for alignment and insertion
- the episode length distribution
- the overall training stability across many iterations


Resume Training
^^^^^^^^^^^^^^^

To continue training from an existing checkpoint:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py \
     --headless \
     --num_envs 4096 \
     --max_iterations 1000 \
     --agent_cfg_path isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml \
     --checkpoint <path/to/checkpoint.pth> \
     nist_assembled_gear_mesh_osc \
     --rl_training_mode

Multi-GPU Training
^^^^^^^^^^^^^^^^^^

Add ``--distributed`` to spread environments across all available GPUs:

.. code-block:: bash

   python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
     isaaclab_arena/scripts/reinforcement_learning/train_rl_games.py \
     --headless \
     --num_envs 4096 \
     --max_iterations 1000 \
     --agent_cfg_path isaaclab_arena_examples/policy/nist_gear_insertion_osc_rl_games.yaml \
     --distributed \
     nist_assembled_gear_mesh_osc \
     --rl_training_mode


Expected Results
^^^^^^^^^^^^^^^^

A successful training run should produce:

- RL Games console output for rollout collection and optimization
- checkpoint files under the RL Games log directory
- progressively improving insertion performance during evaluation

.. image:: ../../../images/nist_gear_insertion_task.gif
   :align: center
   :height: 400px

.. note::

   Training performance depends on hardware, environment configuration, and random seed.
   Because this is a contact-rich insertion task with randomized environment parameters,
   final performance will vary with the exact number of training iterations.
