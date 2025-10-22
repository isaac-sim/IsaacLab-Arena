Creating a New Embodiment
=========================

Learn how to create embodiments using the ``FrankaEmbodiment`` as an example.

Overview
--------

Embodiments define robot configurations. They combine robot assets, control actions, observations, and reset events.

Basic Structure
---------------

All embodiments inherit from ``EmbodimentBase`` and follow this pattern:

.. code-block:: python

    from isaac_arena.embodiments.embodiment_base import EmbodimentBase
    from isaac_arena.assets.register import register_asset

    @register_asset
    class MyRobotEmbodiment(EmbodimentBase):
        name = "my_robot"

        def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
            super().__init__(enable_cameras, initial_pose)
            self.scene_config = MyRobotSceneCfg()
            self.action_config = MyRobotActionsCfg()
            self.observation_config = MyRobotObservationsCfg()
            self.event_config = MyRobotEventCfg()
            self.mimic_env = MyRobotMimicEnv

Key Components
--------------

**1. Scene Configuration**
   - Define robot articulation and supporting assets
   - Configure end-effector frame tracking
   - Use ``{ENV_REGEX_NS}`` pattern for multi-environment support

**2. Action Configuration**
   - Set up control methods (joint control, inverse kinematics)
   - Define gripper or tool control
   - Configure action scaling and limits

**3. Observation Configuration**
   - Specify sensor readings and joint states
   - Group observations by purpose (policy, critic, etc.)
   - Enable camera observations when needed

**4. Event Configuration**
   - Define reset behaviors and initial poses
   - Configure randomization and startup events
   - Handle environment initialization

Example: Franka Embodiment
---------------------------

.. code-block:: python

    @register_asset
    class FrankaEmbodiment(EmbodimentBase):
        name = "franka"

        def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
            super().__init__(enable_cameras, initial_pose)
            self.scene_config = FrankaSceneCfg()
            self.action_config = FrankaActionsCfg()
            self.observation_config = FrankaObservationsCfg()
            self.event_config = FrankaEventCfg()
            self.mimic_env = FrankaMimicEnv

    @configclass
    class FrankaSceneCfg:
        robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector"
            )]
        )

    class FrankaActionsCfg:
        arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand"
        )
        gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"]
        )

Implementation Tips
-------------------

**Configuration Classes**
   - Use ``@configclass`` decorator for scene and observation configs
   - Reference robot assets by consistent naming (typically "robot")
   - Define frame transformers for end-effector tracking

**Multi-Environment Support**
   - Always use ``{ENV_REGEX_NS}`` in prim paths
   - Handle ``env_ids`` parameter in custom methods
   - Support camera integration via ``enable_cameras`` flag

**Asset Integration**
   - Use existing robot configurations from ``isaaclab_assets``
   - Replace ``prim_path`` to match environment naming
   - Configure physics properties and initial states

Usage in Environments
---------------------

Register and use embodiments:

.. code-block:: python

    # Embodiment is automatically registered
    franka = FrankaEmbodiment(enable_cameras=True)

    # Access configurations
    scene_cfg = franka.get_scene_cfg()
    action_cfg = franka.get_action_cfg()
    obs_cfg = franka.get_observation_cfg()

This enables modular robot definitions that can be swapped between tasks and environments.
