from isaac_arena.embodiments.franka.franka_embodiment import (
    FrankaActionsCfg,
    FrankaEmbodiment,
    FrankaEventCfg,
    FrankaObservationsCfg,
)
from isaac_arena.scene.pick_and_place_scene import MugInDrawerKitchenPickAndPlaceScene, PickAndPlaceSceneCfg
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTaskCfg, TerminationsCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class ArenaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: PickAndPlaceSceneCfg = MugInDrawerKitchenPickAndPlaceScene().get_scene_cfg()
    observations: FrankaObservationsCfg = FrankaEmbodiment().get_observation_cfg()
    actions: FrankaActionsCfg = FrankaEmbodiment().get_action_cfg()
    terminations: TerminationsCfg = PickAndPlaceTaskCfg().get_termination_cfg()
    events: FrankaEventCfg = FrankaEmbodiment().get_event_cfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        # TODO(cvolk) The scene needs to hold a robot.
        # --> scene (class InteractiveSceneCfg)
        self.scene.robot = FrankaEmbodiment().get_robot_cfg()

        # TODO(cvolk): Currently lots of helper functions in the mdp (observations.py) expect
        # the "ee_frame" to be defined within a scene object. Update those.
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
