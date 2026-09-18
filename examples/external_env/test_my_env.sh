#!/usr/bin/env bash
# Run the external Newton + SDF environment through the policy runner.
# Must run from the repo root (or anywhere; PYTHONPATH is set below).

cd /home/magengyu/IsaacLab-Arena


PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 24000 \
  --viz kit \
  --presets newton \
  --external_environment_class_path examples.external_env.newton_sdf_env:NewtonSdfEnvironment \
  newton_sdf


PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 5000 \
  --viz kit \
  --external_environment_class_path examples.external_env.franka_table_env:ExternalFrankaTableEnvironment \
  franka_table \
  --object tomato_soup_can


PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type random_action \
  --num_steps 20000 \
  --viz kit \
  --presets newton \
  --print_actions \
  --external_environment_class_path examples.external_env.franka_ABB_env:FrankaAbbFlexiblePackingEnvironment \
  franka_abb_flexible_packing


PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type random_action \
  --num_steps 20000 \
  --viz kit \
  --enable_cameras \
  --print_actions \
  --external_environment_class_path examples.external_env.franka_ABB_env:FrankaAbbFlexiblePackingEnvironment \
  franka_abb_flexible_packing  
