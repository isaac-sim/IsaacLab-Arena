#!/usr/bin/env bash
# Teleoperate the G1 apple-to-plate scene WITHOUT recording (launches Isaac Sim GUI).
# Use it to watch whether the arms actually track the teleop reference motion.
# Run INSIDE the Arena container (enter with:  ./docker/run_docker.sh ).
#
#   ./scripts/teleop.sh                                  # gravity-compensated embodiment (default)
#   ./scripts/teleop.sh g1_wbc_agile_pink                # stock embodiment, for an A/B comparison
#   ./scripts/teleop.sh g1_wbc_agile_pink_gravity_comp keyboard   # override teleop device
#
# Nothing is written to disk: this runs teleop.py, not record_demos.py. Press R in the
# viewport to reset the episode, Ctrl-C to quit.
#
# PREREQ (openxr): start the CloudXR runtime in a SEPARATE container terminal first:
#   python -m isaacteleop.cloudxr        (answer 'y' to the EULA prompt, leave running)
#
# Mirrors record_teleop.sh's environment so what you see here matches what gets recorded:
#   ARM_HOME=zero ./scripts/teleop.sh          # arms straight down instead of the deploy home pose
#   MIRROR_RIGHT=1 ./scripts/teleop.sh         # mirrored scene for right-hand grasps
#   G1_USD_PATH=<other.usd> ./scripts/teleop.sh   # different robot model
set -euo pipefail

EMBODIMENT="${1:-g1_wbc_agile_pink_gravity_comp}"
DEVICE="${2:-openxr}"

# Arm home follows the env default (deploy); override by exporting ARM_HOME=zero.
export ARM_HOME="${ARM_HOME:-deploy}"

# Robot model: prefer the single-file mount from docker/run_docker.sh, then the whole
# robot_menagerie checkout mounted at /menagerie (present on containers started before
# the /robot mount was added). Falls back to the stock Nucleus G1 if neither is there.
if [ -z "${G1_USD_PATH:-}" ]; then
  for candidate in /robot/g1.usda /menagerie/unitree/g1/generated/g1/usd/g1.usda; do
    if [ -f "$candidate" ]; then export G1_USD_PATH="$candidate"; break; fi
  done
fi

cat <<MSG
==================================================================
 Teleop (no recording)
   embodiment: $EMBODIMENT
   device:     $DEVICE
   robot USD:  ${G1_USD_PATH:-<stock Nucleus G1>}
   ARM_HOME:   $ARM_HOME   MIRROR_RIGHT=${MIRROR_RIGHT:-unset}
 -> Isaac Sim window will open. Start the session from the XR tab.
    Watch whether the arms track your controller reference; nothing is saved.
    Press R in the viewport to reset, Ctrl-C to exit.
    The menagerie G1 has BLACK Dex3 hands; all-grey means the stock asset loaded.
==================================================================
MSG

# CloudXR: the host's ~/.cloudxr is bind-mounted at /cloudxr in this container (NOT at
# $HOME), so check both. The runtime must already be running in another terminal, and its
# env must be sourced AFTER the runtime is up and BEFORE Arena starts.
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
exec /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/teleop.py \
  --viz kit --device cpu --enable_cameras \
  g1_apple_to_plate_with_slab \
  --object apple_01_objaverse_robolab --destination clay_plates_hot3d_robolab \
  --embodiment "$EMBODIMENT" --teleop_device "$DEVICE"
