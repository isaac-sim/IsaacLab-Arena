# Isaac Arena Examples

Isaac Arena includes some pre-configured example environments to demonstrate its use.

Right now these environments are
- `PickAndPlaceEnvironment`: A task requiring picking and placing a object in the drawer in a kitchen scene.
- `Gr1OpenMicrowaveEnvironment`: A task requiring opening a microwave door.

Please check the relevant environment files to see what CLI arguments are supported.

Examples are launched with a zero action runner (with some example arguments) like:

```bash
python zero_action_runner pick_and_place --object cracker_box --embodiment gr1
```

or

```bash
python zero_action_runner gr1_open_microwave --object tomato_soup_can
```
