#!/usr/bin/env bash
# Record left-hand teleop seeds for the G1 apple-to-plate task.
# Run INSIDE the Arena container (enter with:  ./docker/run_docker.sh ).
#
# Usage:
#   ./scripts/record_teleop.sh                        # 20 demos, openxr
#   ./scripts/record_teleop.sh 30                     # override demo count
#   ./scripts/record_teleop.sh 30 openxr /datasets/seed_apple_to_plate/my.hdf5
#
# PREREQ: start CloudXR in a SEPARATE container terminal first:
#   python -m isaacteleop.cloudxr   (answer 'y' to EULA, leave running)
set -euo pipefail

NUM="${1:-20}"
DEVICE="${2:-openxr}"
OUT="${3:-/datasets/seed_apple_to_plate/teleop_left${NUM}.hdf5}"

# Robot model: docker/run_docker.sh mounts G1_USD_HOST_PATH as /robot/g1.usda.
# Falls back to the stock Nucleus G1 if the file was not mounted.
export G1_USD_PATH="${G1_USD_PATH:-/robot/g1.usda}"

cat <<MSG
==================================================================
 Teleop recording: LEFT-hand grasp   ARM_HOME=deploy
   demos:   $NUM
   device:  $DEVICE
   output:  $OUT
 -> Isaac Sim window will open. Start the session from the XR tab.
    Grasp the apple with your LEFT controller, place on the plate.
    A demo auto-saves once the apple settles (10 steps).
    Press R to reset an episode. Exits after $NUM demos.
==================================================================
MSG

mkdir -p "$(dirname "$OUT")"

if [ "$DEVICE" = "openxr" ]; then
  CXR_ENV=""
  for c in /cloudxr/run/cloudxr.env "$HOME/.cloudxr/run/cloudxr.env"; do
    if [ -f "$c" ]; then CXR_ENV="$c"; break; fi
  done
  if [ -z "$CXR_ENV" ]; then
    echo "ERROR: CloudXR not initialized. Run first:  python -m isaacteleop.cloudxr"
    exit 1
  fi
  echo "[cloudxr] sourcing $CXR_ENV"
  set +u; source "$CXR_ENV"; set -u
fi

exec /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --viz kit --device cpu --enable_cameras \
  --dataset_file "$OUT" \
  --num_demos "$NUM" --num_success_steps 10 --disable_full_sim_buffer_reset \
  g1_apple_to_plate_with_slab \
  --object apple_01_objaverse_robolab --destination clay_plates_hot3d_robolab \
  --embodiment g1_wbc_agile_pink --teleop_device "$DEVICE"
