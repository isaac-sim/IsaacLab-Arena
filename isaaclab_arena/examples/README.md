# IsaacLab Arena Examples

IsaacLab Arena includes some pre-configured example environments to demonstrate its use.

Right now these environments are
- `KitchenPickAndPlaceEnvironment`: A task requiring picking and placing a object in the drawer in a kitchen scene.
- `GalileoPickAndPlaceEnvironment`: A task requiring picking and placing a object in the drawer in a galileo scene.
- `Gr1OpenMicrowaveEnvironment`: A task requiring opening a microwave door.

Please check the relevant environment files to see what CLI arguments are supported.

Examples are launched with a zero action runner (with some example arguments) like:

```bash
python isaaclab_arena/examples/policy_runner.py --policy_type zero_action kitchen_pick_and_place --object cracker_box --embodiment gr1_joint
```

or

```bash
python isaaclab_arena/examples/policy_runner.py --policy_type zero_action gr1_open_microwave --object tomato_soup_can
```

**NOTE:** CLI arguments are sensitive to order. They must appear in the following order:

```
python isaaclab_arena/examples/policy_runner.py <--global flags> <example app name> <--app specific flags>
```

App specific flags must appear after the first argument with `--` which must be the example app name.
