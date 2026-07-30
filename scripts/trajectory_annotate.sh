#!/usr/bin/env bash
# Auto-annotate Mimic subtask boundaries on the menagerie slab seeds, producing the
# annotated SOURCE dataset that run_generation.sh consumes. --auto replays each seed
# through PHYSICS and keeps it only if the success term + all subtask signals fire, so
# this MUST use the braced slab env (g1_apple_to_plate_with_slab) — the base env would
# fail the replay (base-drift) exactly as it failed the grasp during seed recording.
#   ./run_annotate.sh [input_basename] [output_basename]
set -euo pipefail
IN="${1:-seeds_menagerie_slab}"
OUT="${2:-${IN}_annotated}"
STAMP="2026-07-22"
LOG="$HOME/datasets/seed_apple_to_plate/${OUT}_${STAMP}.log"

# Single solo container (no concurrent boots) so the Kit shader cache can be REUSED.
# Don't host-side `rm -rf` it: the container writes root-owned files there and the
# wipe would hit permission-denied (and abort under `set -e`).
mkdir -p ~/.cache_annotate
docker rm -f annotate_menagerie >/dev/null 2>&1 || true
docker run --rm --name annotate_menagerie \
  --runtime=nvidia --gpus=all --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e G1_USD_PATH=/menagerie/unitree/g1/generated/g1/usd/g1.usda \
  -v ~/Code/IsaacLab-Arena:/workspaces/isaaclab_arena \
  -v ~/Code/robot_menagerie:/menagerie \
  -v ~/datasets:/datasets \
  -v /tmp/Assets:/tmp/Assets \
  -v ~/.cache_annotate:/root/.cache \
  --entrypoint /bin/bash -w /workspaces/isaaclab_arena isaaclab_arena:latest -lc \
  "/isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --headless --enable_cameras --mimic --auto \
     --input_file /datasets/seed_apple_to_plate/${IN}.hdf5 \
     --output_file /datasets/seed_apple_to_plate/${OUT}.hdf5 \
     g1_apple_to_plate_with_slab --embodiment g1_wbc_agile_pink" 2>&1 | tee "$LOG"
