Policy Training
---------------

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Training Command
^^^^^^^^^^^^^^^^

Training uses Arena's bridge to Isaac Lab's unified training script. The ``--external_callback``
argument points to an Arena function that reads the ``--task`` argument, builds the environment,
and registers it with gym before training starts.

.. code-block:: bash

   python isaaclab_arena/scripts/train.py \
     --rl_library rsl_rl \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object \
     --rl_training_mode \
     --num_envs 4096 \
     --max_iterations 2000

.. tip::

   Add ``--viz kit`` to open the GUI and watch training live.

Checkpoints are written to ``logs/rsl_rl/generic_experiment/<timestamp>/``.
The agent configuration is saved alongside as ``params/agent.yaml``,
which the evaluation script uses to reconstruct the policy at inference time.


Overriding Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Hyperparameters come from ``RLPolicyCfg`` in ``isaaclab_arena_examples/policy/base_rsl_rl_policy.py``
and can be overridden with Hydra syntax appended to the training command:

.. code-block:: bash

   # Change network activation function to relu (default: elu)
   agent.policy.activation=relu

   # Adjust the learning rate (default: 0.0001)
   agent.algorithm.learning_rate=0.001

   # Save a checkpoint more frequently (default: every 200 iterations)
   agent.save_interval=500

For example, to train with relu activation and a higher learning rate:

.. code-block:: bash

   python isaaclab_arena/scripts/train.py \
     --rl_library rsl_rl \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object \
     --rl_training_mode \
     --num_envs 4096 \
     --max_iterations 2000 \
     agent.policy.activation=relu \
     agent.algorithm.learning_rate=0.001


Resuming from a Checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^

To resume training from a previously saved checkpoint, use the ``--resume`` flag
together with ``--load_run`` (run folder name) and ``--checkpoint`` (model filename).
Both arguments are optional — when omitted, the most recent run and latest checkpoint
are used automatically.

.. code-block:: bash

   python isaaclab_arena/scripts/train.py \
     --rl_library rsl_rl \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object \
     --rl_training_mode \
     --num_envs 4096 \
     --max_iterations 4000 \
     --resume \
     --load_run <timestamp> \
     --checkpoint model_1999.pt

Replace ``<timestamp>`` with the run folder name under ``logs/rsl_rl/generic_experiment/``.
If ``--load_run`` is omitted, the latest run is selected. If ``--checkpoint`` is omitted,
the latest checkpoint in that run is loaded.


Monitoring Training
^^^^^^^^^^^^^^^^^^^

Launch Tensorboard to monitor progress:

.. code-block:: bash

   python -m tensorboard.main --logdir logs/rsl_rl

Each training iteration prints a summary to the console. This example is from a
short run with ``--max_iterations 20``; the initial training command uses 2000 iterations.
Timings and metric values vary with hardware, configuration, and random seed.

.. code-block:: text

   Learning iteration 3/20

                               Total steps: 393216
                          Steps per second: 20255
                           Collection time: 4.799s
                             Learning time: 0.054s
                           Mean value loss: 0.0028
                       Mean surrogate loss: -0.0017
                         Mean entropy loss: 11.5071
                               Mean reward: 0.75
                       Mean episode length: 92.78
                           Mean action std: 1.02
                  Episode_Reward/joint_vel: -0.0009
        Metrics/object_pose/position_error: 0.2268
             Episode_Reward/lifting_object: 0.1200
       Episode_Reward/object_goal_tracking: 0.0259
     Metrics/object_pose/orientation_error: 3.0806
                Episode_Reward/action_rate: -0.0005
   Episode_Reward/object_goal_tracking_fine_grained: 0.0000
        Episode_Termination/object_dropped: 0.0000
              Episode_Termination/time_out: 0.3371
               Episode_Termination/success: 0.0000
            Episode_Reward/reaching_object: 0.0039
   --------------------------------------------------------------------------------
                            Iteration time: 4.85s
                              Time elapsed: 0:00:19
                                       ETA: 0:01:19


Multi-GPU Training
^^^^^^^^^^^^^^^^^^

Add ``--distributed`` to spread environments across all available GPUs:

.. code-block:: bash

   python isaaclab_arena/scripts/train.py \
     --rl_library rsl_rl \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task lift_object \
     --rl_training_mode \
     --num_envs 4096 \
     --max_iterations 2000 \
     --distributed


Expected Results
^^^^^^^^^^^^^^^^

After 2,000 iterations (~40 minutes on a single GPU with 4096 environments), the trained
policy should reliably grasp and lift objects to commanded target positions.

.. image:: ../../../images/lift_object_rl_task.gif
   :align: center
   :height: 400px

.. note::

   Training performance depends on hardware, environment configuration, and random seed.
   For best results, use a powerful GPU (e.g., RTX 4090, A100, L40).
