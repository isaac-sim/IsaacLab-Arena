### run_curobot

python isaaclab_arena/scripts/curobo/run_droid_v2_tabletop_curobo_pick_place.py \ droid_v3_tabletop_picand_place --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 \ --debug_planner --debug_goal  --grasp_orientation object_yaw --post_place_clearance 0.0 --num_demos 3

### record

python isaaclab_arena/scripts/imitation_learning/record_curobo_demos.py droid_v3_tabletop_pick_and_place \ --grasp_z_offset 0.17 --approach_distance 0.08 --retreat_distance 0.08 --debug_planner --debug_goal \ --grasp_orientation object_yaw --post_place_clearance 0.0 --num_demos 1 --dataset_file /datasets/ \ curobo_v3_statesonly.hdf5


### render

python isaaclab_arena/scripts/imitation_learning/render_demos.py     --dataset_file /datasets/ \ curobo_v3_statesonly.hdf5 --output_dataset /datasets/curobo_v3_with_camera.hdf5     --use_joint_targets \ --enable_cameras droid_v3_tabletop_pick_and_place --embodiment droid_abs_joint_pos \

### hdf5 -> lerobot

python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py      --yaml_file isaaclab_arena_gr00t/lerobot config/droid_joint_position_config.yaml
