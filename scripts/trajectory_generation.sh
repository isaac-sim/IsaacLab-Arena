#!/usr/bin/env bash
# Fan out 4 single-env CPU-physics containers to Mimic-generate the 400-trajectory
# fullrand dataset from the annotated seeds. Each container: distinct seed,
# FULLRAND=1 (actual code DR: apple +/-7.5cm rejection-sampled off plate,
# plate +/-5cm, base +/-1.5cm + yaw +/-1.5deg, arms +/-2.25deg; pose-only, no visual DR),
# 100 trials -> 100 successful demos (generation_guarantee). CPU physics so 4 fit on
# one GPU (only camera rendering uses the GPU).
# Uses the BRACED slab env (g1_apple_to_plate_with_slab): the menagerie G1 hand only
# lifts when the pelvis is braced (base env = 0/27), so generation must match the env
# the seeds were recorded + annotated in.
#   ./run_generation.sh [annotated_hdf5] [trials_per_container]
set -eu
INPUT="${1:-$HOME/datasets/seed_apple_to_plate/seeds_menagerie_slab_annotated.hdf5}"
TRIALS="${2:-100}"
BN="$(basename "$INPUT")"
OUTDIR="$HOME/datasets/isaaclab_arena/static_apple_tutorial"
mkdir -p "$OUTDIR"
test -f "$INPUT" || { echo "ERROR: annotated input $INPUT not found"; exit 1; }

# container index -> seed (drives BOTH the reset-randomization RNG via --seed and
# the Mimic datagen stochasticity via MIMIC_SEED, so each container is distinct).
seeds=(101 202 303 404)
for i in 0 1 2 3; do
  n=$((i+1)); Si=${seeds[$i]}
  # Per-container isolated cache (satisfies the concurrent-boot requirement — the 4
  # containers never SHARE a Kit shader cache). REUSE it rather than host-side `rm -rf`:
  # the container writes root-owned files here, so the wipe would hit permission-denied
  # and abort under `set -eu`. Content-hashed shader kernels are safe to reuse.
  mkdir -p "$HOME/.cache_gen_$n"
  docker rm -f gen_$n >/dev/null 2>&1 || true
  docker run -d --name gen_$n --runtime=nvidia --gpus=all --ipc=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -e G1_USD_PATH=/menagerie/unitree/g1/generated/g1/usd/g1.usda \
    -e FULLRAND=1 -e MIMIC_SEED="$Si" \
    -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=6 -e MKL_NUM_THREADS=6 \
    -v ~/Code/IsaacLab-Arena:/workspaces/isaaclab_arena \
    -v ~/Code/robot_menagerie:/menagerie -v ~/datasets:/datasets \
    -v /tmp/Assets:/tmp/Assets \
    -v "$HOME/.cache_gen_$n":/root/.cache \
    --entrypoint /bin/bash -w /workspaces/isaaclab_arena isaaclab_arena:latest -lc \
    "/isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
       --headless --enable_cameras --mimic --device cpu --seed $Si \
       --generation_num_trials $TRIALS \
       --input_file /datasets/seed_apple_to_plate/$BN \
       --output_file /datasets/isaaclab_arena/static_apple_tutorial/gen400_menagerie_part_$n.hdf5 \
       g1_apple_to_plate_with_slab --embodiment g1_wbc_agile_pink" >/dev/null
  echo "launched gen_$n (seed $Si) -> gen400_menagerie_part_$n.hdf5"
done
echo "4 generation containers up. Monitor: docker logs gen_<n>  |  ls -la $OUTDIR/gen400_menagerie_part_*.hdf5"
