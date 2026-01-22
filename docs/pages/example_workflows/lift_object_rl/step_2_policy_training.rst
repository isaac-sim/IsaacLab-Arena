Policy Training
---------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

This page covers training a reinforcement learning policy for the lift object task using RSL-RL.


Training Overview
^^^^^^^^^^^^^^^^^

The lift object environment is designed for reinforcement learning. We train a policy that learns to
control the Franka robot arm to grasp and lift objects to a target height. The training uses the
PPO (Proximal Policy Optimization) algorithm via the RSL-RL library.

The RL task environment additionallyprovides:

* **Observations**: Robot joint positions, velocities, and object state
* **Commands**: Target object pose for the robot arm
* **Rewards**: Based on distance to object, grasp success, and lifting height


Training Command
^^^^^^^^^^^^^^^^

To start training, run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train.py \
     --env_spacing 30.0 \
     --num_envs 512 \
     --max_iterations 12000 \
     --headless
     lift_object \
     --embodiment franka \
     --object dex_cube

**Command Arguments:**

* ``lift_object``: The environment name
* ``--embodiment franka``: The robot to use (Franka Panda arm)
* ``--object dex_cube``: The object to lift (can also use other objects)
* ``--num_envs 512``: Number of parallel environments (adjust based on GPU memory)
* ``--env_spacing 30.0``: Spacing between environments
* ``--max_iterations 12000``: Maximum number of iterations
* ``--headless``: Run without GUI for faster training

Training will run for the default number of iterations (typically around 1000-2000) and save
checkpoints periodically in the ``logs/rsl_rl/lift_object/`` directory.


Training Configuration
^^^^^^^^^^^^^^^^^^^^^^

The default training configuration is located at:

``isaaclab_arena/policy/rl_policy/generic_policy.json``

You can customize training hyperparameters by creating a custom configuration file and
passing it with the ``--agent_cfg_path`` argument:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train.py \
     --num_envs 512 \
     --headless \
     --agent_cfg_path path/to/custom_config.json
     lift_object \
     --embodiment franka \
     --object dex_cube


**Key hyperparameters:**

* Learning rate
* Number of steps per iteration
* Mini-batch size
* Number of epochs per iteration
* Discount factor (gamma)


Monitoring Training
^^^^^^^^^^^^^^^^^^^

Training progress is logged to TensorBoard. To monitor training:

.. code-block:: bash

   tensorboard --logdir logs/rsl_rl/lift_object

Then open your browser to ``http://localhost:6006`` to view training metrics including:

* Episode rewards
* Episode lengths
* Policy loss
* Value loss
* Success rate


Multi-GPU Training
^^^^^^^^^^^^^^^^^^

For faster training on multi-GPU systems, use the ``--distributed`` flag:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/train.py \
     --num_envs 512 \
     --headless \
     --distributed \
     lift_object \
     --embodiment franka \
     --object dex_cube

This will automatically distribute the training across available GPUs.


Expected Results
^^^^^^^^^^^^^^^^

Training typically takes 24 hours on a modern GPU (e.g., RTX 4090, L40S) for the policy to learn
the lifting task. You should see:

* Episode rewards increasing over time
* Success rate improving to 80-90% after convergence
* The robot learning to approach, grasp, and lift objects consistently

The trained policy checkpoint will be saved in the experiment log directory and can be used
for evaluation and deployment.
