# IsaacLab Arena Examples

IsaacLab Arena includes some pre-configured example environments to demonstrate its use.

Right now these environments are
- `KitchenPickAndPlaceEnvironment`: A task requiring picking and placing a object in the drawer in a kitchen scene.
- `GalileoPickAndPlaceEnvironment`: A task requiring picking and placing a object in the drawer in a galileo scene.
- `Gr1OpenMicrowaveEnvironment`: A task requiring opening a microwave door.

Please check the relevant environment files to see which configuration fields each environment supports.

Examples are launched as an Arena Experiment with the zero action policy. The Experiment
[`example_environments_zero_action_experiment.yaml`](../../isaaclab_arena_environments/experiment_configs/examples/example_environments_zero_action_experiment.yaml)
has one Run per example. Each Run selects an environment by its registered name in
`environment.type` and sets some example fields below `environment`:

```yaml
runs:
  kitchen_pick_and_place:
    environment:
      type: kitchen_pick_and_place
      object: cracker_box
      embodiment: gr1_joint

  gr1_open_microwave:
    environment:
      type: gr1_open_microwave
      object: tomato_soup_can
```

Both Runs share the `zero_action` policy and stop after 100 steps (`rollout_limit.num_steps`).
Run them from the repository root:

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/experiment_configs/examples/example_environments_zero_action_experiment.yaml
```

To change a field of one Run, add a `runs.<run name>.environment.<field>=<value>` override:

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/experiment_configs/examples/example_environments_zero_action_experiment.yaml \
  runs.kitchen_pick_and_place.environment.object=tomato_soup_can
```
