Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained RSL-RL policy in closed-loop
and evaluating it in the Lift Object environment.

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

Once inside the container, set the models directory if you plan to download pre-trained checkpoints:

.. code:: bash

    export MODELS_DIR=models/isaaclab_arena/reinforcement_learning
    mkdir -p $MODELS_DIR

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_2_policy_training>` and have a trained checkpoint available,
or you can download a pre-trained checkpoint as described below.

.. dropdown:: Download Pre-trained Model (skip preceding steps)
   :animate: fade-in

   These commands can be used to download a pre-trained RSL-RL policy checkpoint,
   such that the preceding training step can be skipped.

   .. code-block:: bash

      hf download \
         nvidia/IsaacLab-Arena-Lift-Object-RL \
         model_11999.pt \
         --local-dir $MODELS_DIR/lift_object_checkpoint

   After downloading, you can use the checkpoint at:

   ``$MODELS_DIR/lift_object_checkpoint/model_11999.pt``

   Replace checkpoint paths in the examples below with this path to evaluate the pre-trained model.


Evaluation Methods
^^^^^^^^^^^^^^^^^^

Isaac Lab Arena provides multiple ways to evaluate trained RL policies:

1. **Quick Visualization (play.py)**: Fast visual inspection of policy behavior
2. **Single Environment Evaluation (policy_runner.py)**: Detailed evaluation with metrics
3. **Parallel Environment Evaluation (policy_runner.py)**: Large-scale statistical evaluation
4. **Batch Evaluation (eval_runner.py)**: Automated evaluation of multiple checkpoints


Method 1: Quick Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``play.py`` script provides the fastest way to visually inspect your trained policy.
This is useful for debugging and quick quality checks.

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/play.py \
     --env_spacing 30.0 \
     --num_envs 16 \
     --checkpoint logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt \
     lift_object

**Key Features:**

- Fast startup with GUI enabled by default
- Visualizes policy rollouts in real-time
- No metrics computation (pure visualization)
- Useful for debugging policy behavior

**Command Arguments:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--env_spacing 30.0``
     - Larger spacing for visualization (avoids visual clutter)
   * - ``--num_envs 16``
     - Number of parallel environments to visualize
   * - ``--checkpoint <path>``
     - Path to the trained model checkpoint (.pt file)
   * - ``lift_object``
     - Environment name (must be last)

You should see multiple Franka robots simultaneously attempting to lift objects to various target positions.


Method 2: Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``policy_runner.py`` provides comprehensive evaluation with task-specific metrics.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_steps 1000 \
     --checkpoint_path logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt \
     lift_object \
     --rl_training_mode False

.. note::

   If you downloaded the pre-trained model from Hugging Face, replace the checkpoint path:

   ``--checkpoint_path $MODELS_DIR/lift_object_checkpoint/model_11999.pt``

**Important: Argument Order**

Policy-specific arguments (``--policy_type``, ``--checkpoint_path``, etc.) must come **before** the environment name.
Environment-specific arguments (``--rl_training_mode``, ``--object``, etc.) must come **after** the environment name.

**Command Breakdown:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--policy_type rsl_rl``
     - Policy type to load (RSL-RL trained policy)
   * - ``--num_steps 1000``
     - Total simulation steps to run
   * - ``--checkpoint_path <path>``
     - Path to the model checkpoint
   * - ``lift_object``
     - Environment name
   * - ``--rl_training_mode False``
     - Enable success termination for evaluation

**Expected Output:**

At the end of evaluation, you should see metrics similar to:

.. code-block:: text

   Metrics: {'success_rate': 0.85, 'num_episodes': 12}

This indicates that 85% of episodes successfully lifted the object to the target position,
across 12 completed episodes in 1000 steps.


Method 3: Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more statistically significant results, evaluate across many parallel environments:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --num_steps 5000 \
     --num_envs 64 \
     --checkpoint_path logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt \
     --headless \
     lift_object \
     --rl_training_mode False

**Additional Arguments:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--num_envs 64``
     - Run 64 parallel environments simultaneously
   * - ``--headless``
     - Run without GUI for faster evaluation
   * - ``--num_steps 5000``
     - More steps for more episodes

**Expected Output:**

.. code-block:: text

   Metrics: {'success_rate': 0.83, 'num_episodes': 156}

Running more environments and steps provides better statistical estimates of policy performance.


Method 4: Batch Evaluation with JSON Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For systematic evaluation of multiple checkpoints or hyperparameter sweeps, use ``eval_runner.py``
with a JSON configuration file.

**1. Create Evaluation Configuration**

Create a file ``eval_config.json``:

.. code-block:: json

   {
     "policy_runner_args": {
       "policy_type": "rsl_rl",
       "num_steps": 5000,
       "num_envs": 64,
       "headless": true
     },
     "evaluations": [
       {
         "checkpoint_path": "logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_5999.pt",
         "environment": "lift_object",
         "environment_args": {
           "rl_training_mode": false
         }
       },
       {
         "checkpoint_path": "logs/rsl_rl/generic_experiment/2026-01-28_17-26-10/model_11999.pt",
         "environment": "lift_object",
         "environment_args": {
           "rl_training_mode": false
         }
       }
     ]
   }

**2. Run Batch Evaluation**

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py --config eval_config.json

This will automatically evaluate all checkpoints listed in the configuration and output
a summary of metrics for each.

**Expected Output:**

.. code-block:: text

   Evaluating checkpoint 1/2: model_5999.pt
   Metrics: {'success_rate': 0.72, 'num_episodes': 152}

   Evaluating checkpoint 2/2: model_11999.pt
   Metrics: {'success_rate': 0.85, 'num_episodes': 156}

   Summary:
   ========================================
   model_5999.pt  | Success: 72% | Episodes: 152
   model_11999.pt | Success: 85% | Episodes: 156


Understanding the Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Lift Object task reports the following metrics:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Metric
     - Description
   * - ``success_rate``
     - Fraction of episodes where object reached target position within tolerance
   * - ``num_episodes``
     - Total number of episodes completed during evaluation

A well-trained policy should achieve:

- **Success rate**: 70-90% (depends on target range difficulty)
- **Consistent performance**: Success rate stable across multiple evaluation runs


Troubleshooting
^^^^^^^^^^^^^^^

**Issue: Low success rate (<50%)**

- Increase training iterations (try 20,000+)
- Check reward configuration in task definition
- Verify command sampling ranges are reasonable
- Try different random seeds

**Issue: Policy gets stuck or drops object**

- Ensure object mass and friction are reasonable
- Check gripper force limits
- Visualize with ``play.py`` to diagnose behavior
- Review episode recordings if ``--video`` was enabled during training

**Issue: "Checkpoint not found" error**

- Verify checkpoint path is correct
- Use absolute paths if relative paths fail
- Check that training completed and saved checkpoints

.. note::

   When running evaluation, always set ``--rl_training_mode False`` to enable success termination.
   During training, this flag is ``True`` by default to prevent early episode termination.
