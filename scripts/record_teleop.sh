#!/usr/bin/env bash
# Record a teleop demo dataset for the G1 apple-to-plate task (launches Isaac Sim GUI).
# Run INSIDE the Arena container (enter with:  ./docker/run_docker.sh ).
#
#   ./record_teleop.sh left            # 20 left-hand grasps (apple on the left, normal scene)
#   ./record_teleop.sh right           # 20 right-hand grasps (apple on the right, mirrored)
#   ./record_teleop.sh left  30        # override demo count
#   ./record_teleop.sh right 20 keyboard              # override teleop device
#   ./record_teleop.sh left  20 openxr /datasets/seed_apple_to_plate/my.hdf5   # override output
#
# PREREQ (openxr): start the CloudXR runtime in a SEPARATE container terminal first:
#   python -m isaacteleop.cloudxr        (answer 'y' to the EULA prompt, leave running)
#
# Arms initialize at the env default home (deploy = real-robot bring-up pose).
#   Override with:  ARM_HOME=zero ./record_teleop.sh left    (all arm joints 0, arms straight down)
# Robot model defaults to the robot_menagerie G1 (G1_USD_PATH export below); override with
#   G1_USD_PATH=<other.usd> ./record_teleop.sh left    to record with a different model.
set -euo pipefail

SIDE="${1:-}"; NUM="${2:-20}"; DEVICE="${3:-openxr}"
case "$SIDE" in
  left)  HAND="LEFT";  DEF_OUT="/datasets/seed_apple_to_plate/teleop_left${NUM}.hdf5";  unset MIRROR_RIGHT 2>/dev/null || true ;;
  right) HAND="RIGHT"; DEF_OUT="/datasets/seed_apple_to_plate/teleop_right${NUM}.hdf5"; export MIRROR_RIGHT=1 ;;
  *) echo "Usage: $0 <left|right> [num_demos=20] [device=openxr] [output_file]"; exit 1 ;;
esac
OUT="${4:-$DEF_OUT}"

# Arm home follows the env default (deploy); override by exporting ARM_HOME=zero.
export ARM_HOME="${ARM_HOME:-deploy}"

# Robot model: docker/run_docker.sh mounts G1_USD_HOST_PATH as /robot/g1.usda.
# Falls back to the stock Nucleus G1 if the file was not mounted.
export G1_USD_PATH="${G1_USD_PATH:-/robot/g1.usda}"

# DR level: default (no var) = FULLRAND ±7.5cm apple / ±5cm plate.
# Override with MEDRAND=1 for tighter ±2cm or ZERODR=1 for nominal only.

cat <<MSG
==================================================================
 Teleop recording: ${HAND}-hand grasp   (MIRROR_RIGHT=${MIRROR_RIGHT:-unset}, ARM_HOME=${ARM_HOME})
   demos:   $NUM
   device:  $DEVICE
   output:  $OUT
 -> Isaac Sim window will open. Start the session from the XR tab.
    Teleoperate BOTH arms; grasp the apple with your ${HAND} controller.
    A demo auto-saves once the apple settles in the plate (10 steps).
    Press R in the viewport to reset an episode. Exits after $NUM demos.
==================================================================
MSG

mkdir -p "$(dirname "$OUT")"

# CloudXR: the host's ~/.cloudxr is bind-mounted at /cloudxr in this container (NOT at
# $HOME), so check both. The runtime must already be running in another terminal, and its
# env must be sourced AFTER the runtime is up and BEFORE Arena starts (docs step_2).
if [ "$DEVICE" = "openxr" ]; then
  CXR_ENV=""
  for c in /cloudxr/run/cloudxr.env "$HOME/.cloudxr/run/cloudxr.env"; do
    if [ -f "$c" ]; then CXR_ENV="$c"; break; fi
  done
  if [ -z "$CXR_ENV" ]; then
    echo "ERROR: CloudXR not initialized. In a SEPARATE container terminal, run FIRST:"
    echo "         python -m isaacteleop.cloudxr     (answer 'y' to the EULA prompt)"
    echo "       then re-run this script."
    exit 1
  fi
  echo "[cloudxr] sourcing $CXR_ENV"
  set +u; source "$CXR_ENV"; set -u
fi

# --viz kit => launch the Isaac Sim GUI (omitting it would run headless).
exec /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --viz kit --device cpu --enable_cameras \
  --dataset_file "$OUT" \
  --num_demos "$NUM" --num_success_steps 10 --disable_full_sim_buffer_reset \
  g1_apple_to_plate_with_slab \
  --object apple_01_objaverse_robolab --destination clay_plates_hot3d_robolab \
  --embodiment g1_wbc_agile_pink --teleop_device "$DEVICE"
