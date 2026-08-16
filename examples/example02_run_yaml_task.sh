# viz mode
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type zero_action \
  --num_steps 200 \
  --env_graph_spec_yaml \
  isaaclab_arena_environments/robolab/tasks/bbq_sauce_in_bin.yaml
