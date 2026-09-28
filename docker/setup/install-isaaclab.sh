#!/bin/bash
# Install Isaac Lab and its dependencies.

set -euo pipefail

# Point Isaac Lab at the Isaac Sim installation already provided by the base image.
ln -s /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
# Run Isaac Lab's own installer to register its packages and install their dependencies.
"${ISAACLAB_PATH}/isaaclab.sh" -i
