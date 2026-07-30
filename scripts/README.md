# G1 Apple-to-Plate Pipeline

```bash
# 1. Record teleop seeds (inside Arena container, CloudXR running in a second terminal)
./scripts/record_teleop.sh left 30 openxr /datasets/seed_apple_to_plate/teleop_left30.hdf5

# 2. Annotate seeds with Mimic subtask boundaries
./scripts/trajectory_annotate.sh teleop_left30 teleop_left30_annotated

# 3. Generate 400 Mimic trajectories (4 containers × 100 trials)
./scripts/trajectory_generation.sh /home/dimos/datasets/seed_apple_to_plate/teleop_left30_annotated.hdf5

# 4. Convert to LeRobot, upload to swift, submit OSMO finetune — see osmo/finetune.yaml
```
