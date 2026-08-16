#!/usr/bin/env bash
# Run the external env defined in my_env.py through the policy runner.
# Must run from the repo root (or anywhere; PYTHONPATH is set below).

cd "$(dirname "$0")/../.."

PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 50 \
  --headless \
  --external_environment_class_path examples.external_env.my_env:ExternalFrankaTableEnvironment \
  franka_table \
  --object tomato_soup_can
