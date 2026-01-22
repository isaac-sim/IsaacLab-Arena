Policy Evaluation
-----------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

This page covers evaluating a trained reinforcement learning policy for the lift object task.


Evaluation Overview
^^^^^^^^^^^^^^^^^^^

After training a policy using RSL-RL, you can evaluate its performance by running it in the
lift object environment. Isaac Lab Arena provides multiple evaluation approaches:

1. **Play Script** (``play.py``) - Direct RSL-RL policy execution with visualization and video recording
2. **Policy Evaluator** (``policy_runner.py``) - Unified evaluation framework with metrics computation
3. **Batch Evaluation** (``eval_runner.py``) - JSON-based configuration for systematic evaluation

All approaches control the Franka robot arm to grasp and lift objects to target positions.

**When to Use Each Approach:**

* Use ``play.py`` for quick visualization, debugging, and creating videos of your trained policy
* Use ``policy_runner.py`` when you need metrics computation or want to evaluate different policy types
* Use ``eval_runner.py`` for systematic evaluation of multiple checkpoints or configurations

.. note::

   For ``policy_runner.py``, the argument order matters: policy arguments (like ``--policy_type``,
   ``--checkpoint_path``) must come **before** the environment name, while environment arguments
   (like ``--embodiment``, ``--object``) must come **after** the environment name.


Visualize Trained Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the trained policy in real-time:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/play.py lift_object \
     --embodiment franka \
     --object dex_cube \
     --num_envs 1 \
     --checkpoint logs/rsl_rl/lift_object/model_1000.pt

**Command Arguments:**

* ``lift_object``: The environment name
* ``--embodiment franka``: Robot to use
* ``--object dex_cube``: Object to lift
* ``--num_envs 1``: Single environment for visualization
* ``--checkpoint``: Path to trained model checkpoint

This will open a viewer window showing the robot executing the trained policy.


Record Video
^^^^^^^^^^^^

To record a video of the policy execution:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/play.py lift_object \
     --embodiment franka \
     --object dex_cube \
     --num_envs 1 \
     --checkpoint logs/rsl_rl/lift_object/model_1000.pt \
     --video \
     --video_length 200

The video will be saved to the log directory under ``videos/play/``.


Evaluate Performance
^^^^^^^^^^^^^^^^^^^^

To evaluate policy performance across multiple parallel environments:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/play.py lift_object \
     --embodiment franka \
     --object dex_cube \
     --num_envs 128 \
     --checkpoint logs/rsl_rl/lift_object/model_1000.pt \
     --headless

Running with more environments provides better statistics for success rate evaluation.


Using the Policy Evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Arena provides a unified policy evaluation framework that works with different policy
types and can compute metrics. To evaluate the RSL-RL policy using the policy evaluator:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type rsl_rl \
     --checkpoint_path logs/rsl_rl/lift_object/model_1000.pt \
     --num_steps 500 \
     lift_object \
     --embodiment franka \
     --object dex_cube \
     --num_envs 16

**Argument Order (Important!):**

All policy-specific arguments must come **before** the environment name, and environment-specific
arguments come **after** the environment name:

.. code-block:: text

   policy_runner.py [POLICY_ARGS] ENVIRONMENT_NAME [ENVIRONMENT_ARGS]
                    ↑                               ↑
                    Before                          After

* **Before** ``lift_object``: ``--policy_type``, ``--checkpoint_path``, ``--num_steps``, ``--headless``, etc.
* **After** ``lift_object``: ``--embodiment``, ``--object``, ``--num_envs``, etc.

**Policy Evaluator Arguments:**

* ``--policy_type rsl_rl``: Specifies the RSL-RL policy type
* ``--checkpoint_path``: Path to the trained checkpoint
* ``--num_steps``: Number of steps to run the evaluation
* ``lift_object``: Environment name (separates policy args from environment args)
* ``--embodiment``, ``--object``, ``--num_envs``: Environment-specific arguments

**Advantages of the Policy Evaluator:**

* **Unified Interface**: Same evaluation script works for RL policies, replay policies, and custom policies
* **Metrics Computation**: Automatically computes and reports registered metrics
* **Flexible Configuration**: Can be configured via CLI or JSON files for batch evaluation

If metrics are registered for the task, they will be automatically computed and displayed at the
end of the evaluation run.


Batch Evaluation with JSON Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For evaluating multiple checkpoints or configurations systematically, you can use JSON-based
configuration with the eval runner. Create a jobs configuration file:

.. code-block:: json

   {
     "jobs": [
       {
         "name": "eval_lift_cube_checkpoint_500",
         "policy_type": "rsl_rl",
         "policy_config_dict": {
           "checkpoint_path": "logs/rsl_rl/lift_object/model_500.pt",
           "agent_cfg_path": "isaaclab_arena/policy/rl_policy/generic_policy.json",
           "device": "cuda:0",
           "clip_actions": true
         },
         "arena_env_args": ["lift_object", "--embodiment", "franka", "--object", "dex_cube"],
         "num_steps": 500
       },
       {
         "name": "eval_lift_cube_checkpoint_1000",
         "policy_type": "rsl_rl",
         "policy_config_dict": {
           "checkpoint_path": "logs/rsl_rl/lift_object/model_1000.pt",
           "agent_cfg_path": "isaaclab_arena/policy/rl_policy/generic_policy.json",
           "device": "cuda:0",
           "clip_actions": true
         },
         "arena_env_args": ["lift_object", "--embodiment", "franka", "--object", "dex_cube"],
         "num_steps": 500
       }
     ]
   }

Save this as ``eval_jobs.json`` and run:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --jobs_config eval_jobs.json \
     --num_steps 500 \
     --headless

This approach is useful for:

* Comparing multiple training checkpoints
* Evaluating across different objects or robot configurations
* Running systematic ablation studies
* Generating comprehensive evaluation reports


Using Pre-trained Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't have a trained checkpoint, you can download a pre-trained model from Hugging Face:

.. code-block:: bash

   # Set models directory
   export MODELS_DIR=/models/isaaclab_arena/lift_object_rl_tutorial
   mkdir -p $MODELS_DIR

   # Download pre-trained checkpoint (example - check Hugging Face for actual paths)
   hf download \
       nvidia/IsaacLab-Arena-Checkpoints \
       lift_object/model_final.pt \
       --repo-type model \
       --local-dir $MODELS_DIR

Then evaluate using the downloaded checkpoint:

.. code-block:: bash

   python isaaclab_arena/scripts/reinforcement_learning/play.py lift_object \
     --embodiment franka \
     --object dex_cube \
     --num_envs 1 \
     --checkpoint $MODELS_DIR/lift_object/model_final.pt


Checkpoint Selection
^^^^^^^^^^^^^^^^^^^^

By default, the play script loads the latest checkpoint from the training log directory.
You can also specify which checkpoint to load:

.. code-block:: bash

   # Load specific checkpoint by path
   --checkpoint logs/rsl_rl/lift_object/2026-01-20_10-30-00/model_1000.pt

   # Load latest checkpoint from a specific run
   --load_run 2026-01-20_10-30-00

   # Load specific iteration from latest run
   --load_checkpoint model_500.pt


Expected Results
^^^^^^^^^^^^^^^^

A well-trained policy should demonstrate:

* **Approach Phase**: Robot arm moves toward the object smoothly
* **Grasp Phase**: Gripper closes around the object securely
* **Lift Phase**: Object is lifted above the minimum height threshold (0.04m)
* **Hold Phase**: Object is maintained at the target height

Success rate should be above 80-90% for a converged policy. If the policy performs poorly,
consider:

* Training for more iterations
* Adjusting reward weights
* Increasing the number of parallel environments during training
* Checking if the object/robot configuration matches training setup


Troubleshooting
^^^^^^^^^^^^^^^

**Command line argument errors with policy_runner.py:**

- Ensure policy arguments (``--policy_type``, ``--checkpoint_path``, ``--num_steps``) come **before** the environment name
- Environment arguments (``--embodiment``, ``--object``, ``--num_envs``) must come **after** the environment name
- Incorrect order will result in parsing errors

**Policy doesn't move the robot:**

- Check that the checkpoint path is correct
- Verify the agent configuration matches the training setup
- Ensure the environment name matches: ``lift_object``

**Poor performance despite training:**

- The policy may need more training iterations
- Try different random seeds for evaluation
- Verify observation normalization is working correctly

**Simulation runs slowly:**

- Use ``--headless`` flag to disable rendering
- Reduce ``--num_envs`` if running out of GPU memory
- Check GPU utilization with ``nvidia-smi``
