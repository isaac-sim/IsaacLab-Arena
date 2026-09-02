#!/usr/bin/env bash
# Mimic-generate the fullrand dataset from annotated seeds: 4 CPU-physics containers
# (so 4 share one GPU), FULLRAND=1 + visual DR. Table height is static per process, so
# each runs PASSES passes at different heights (table_dz). Needs the BRACED slab env,
# matching how the seeds were annotated.
#   ./trajectory_generation.sh [annotated_hdf5] [trials_per_container] [passes]
set -eu
INPUT="${1:-$HOME/datasets/seed_apple_to_plate/seeds_menagerie_slab_annotated.hdf5}"
TRIALS="${2:-100}"
PASSES="${3:-4}"
BN="$(basename "$INPUT")"
OUTDIR="$HOME/datasets/isaaclab_arena/static_apple_tutorial"
mkdir -p "$OUTDIR"
test -f "$INPUT" || { echo "ERROR: annotated input $INPUT not found"; exit 1; }

# Change per run: a reused prefix overwrites the part files.
PREFIX="${PREFIX:-gen400_menagerie}"

STAGGER="${STAGGER:-240}"

# Empty = 24-level sRGB ramp; one value pins the tint (0.272 = nominal).
TABLE_SRGB_LEVELS="${TABLE_SRGB_LEVELS:-}"
PER_PASS=$(( TRIALS / PASSES ))
test "$PER_PASS" -ge 1 || { echo "ERROR: trials ($TRIALS) < passes ($PASSES)"; exit 1; }

# Base seed per container; each pass adds its index.
seeds=(101 202 303 404)
# Table-height offsets (m) per pass: 16 values over +/-0.05, no gaps.
table_dz=(
  "-0.0500 -0.0433 -0.0367 -0.0300"
  "-0.0233 -0.0167 -0.0100 -0.0033"
  "0.0033 0.0100 0.0167 0.0233"
  "0.0300 0.0367 0.0433 0.0500"
)

for i in 0 1 2 3; do
  n=$((i+1)); Si=${seeds[$i]}
  DZ="$(echo "${table_dz[$i]}" | cut -d' ' -f1-"$PASSES")"
  # Per-container shader cache, never shared. Reused not wiped (root-owned, set -eu).
  mkdir -p "$HOME/.cache_gen_$n"
  docker rm -f gen_$n >/dev/null 2>&1 || true
  docker run -d --name gen_$n --runtime=nvidia --gpus=all --ipc=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
    -e G1_USD_PATH=/menagerie/unitree/g1/generated/g1/usd/g1.usda \
    -e FULLRAND=1 \
    -e TABLE_SRGB_LEVELS="$TABLE_SRGB_LEVELS" \
    -e OMP_NUM_THREADS=6 -e OPENBLAS_NUM_THREADS=6 -e MKL_NUM_THREADS=6 \
    -v ~/Code/IsaacLab-Arena:/workspaces/isaaclab_arena \
    -v ~/Code/robot_menagerie:/menagerie -v ~/datasets:/datasets \
    -v /tmp/Assets:/tmp/Assets \
    -v "$HOME/.cache_gen_$n":/root/.cache \
    --entrypoint /bin/bash -w /workspaces/isaaclab_arena isaaclab_arena:latest -lc \
    "set -e
     k=0
     for dz in $DZ; do
       k=\$((k+1))
       echo \"=== gen_$n pass \$k/$PASSES  table_dz=\$dz ===\"
       TABLE_HEIGHT_OFFSET_M=\$dz MIMIC_SEED=\$(( $Si + k )) \
       /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
         --headless --enable_cameras --mimic --device cpu --seed \$(( $Si + k )) \
         --generation_num_trials $PER_PASS \
         --input_file /datasets/seed_apple_to_plate/$BN \
         --output_file /datasets/isaaclab_arena/static_apple_tutorial/${PREFIX}_part_${n}_p\${k}.hdf5 \
         g1_apple_to_plate_with_slab --embodiment g1_wbc_agile_pink
     done" >/dev/null
  echo "launched gen_$n (seed $Si, $PASSES passes x $PER_PASS trials, table_dz:$DZ)"
  # Concurrent boots wedge on the single-threaded scene parse.
  if [ "$n" -lt 4 ]; then
    echo "  ... waiting ${STAGGER}s before next launch"
    sleep "$STAGGER"
  fi
done
echo
echo "4 containers up, $PASSES passes each -> $((4 * PASSES)) part files, $((4 * PER_PASS * PASSES)) demos."
echo "Monitor: docker logs -f gen_1  |  ls -la $OUTDIR/${PREFIX}_part_*_p*.hdf5"
