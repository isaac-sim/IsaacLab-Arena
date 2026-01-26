"""Default configuration values for Isaac Lab Mimic data generation in Isaac Lab Arena."""

MIMIC_DATAGEN_CONFIG_DEFAULTS = {
    "generation_guarantee": True,
    "generation_keep_failed": False,
    "generation_num_trials": 100,
    "generation_select_src_per_subtask": False,
    "generation_select_src_per_arm": False,
    "generation_relative": False,
    "generation_joint_pos": False,
    "generation_transform_first_robot_pose": False,
    "generation_interpolate_from_last_target_pose": True,
    "max_num_failures": 25,
    "seed": 1,
}