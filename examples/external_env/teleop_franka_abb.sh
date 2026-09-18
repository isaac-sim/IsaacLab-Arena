#!/usr/bin/env bash
# Run keyboard teleoperation for the Franka ABB flexible-packing environment.

cd /home/magengyu/IsaacLab-Arena

PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python \
  submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
  --viz kit \
  --teleop_device keyboard \
  --presets newton \
  --enable_cameras \
  --arena_teleop_device keyboard \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --external_environment_class_path examples.external_env.franka_ABB_env:FrankaAbbFlexiblePackingEnvironment \
  --task franka_abb_flexible_packing
