#!/bin/bash
# Install Isaac Lab and its dependencies.

set -euo pipefail

# Point Isaac Lab at the Isaac Sim installation already provided by the base image.
ln -s /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
# The legacy installer performs several independent pip resolutions. Constrain
# each pass to the versions selected by the 3.0 release so later dependency
# installs cannot replace the CUDA 13 torch stack installed by the first pass.
cat >/tmp/isaaclab-install-constraints.txt <<'EOF'
torch==2.11.0
torchvision==0.26.0
warp-lang==1.17.0
newton==1.6.1
mujoco~=3.12.0
mujoco-warp~=3.12.0
numpy==2.3.1
packaging==26.0
urllib3>=2.8.0
websockets==16.1.1
EOF
export PIP_CONSTRAINT=/tmp/isaaclab-install-constraints.txt
# Install the core through Lab's compatibility installer. Isaac Sim 6.2's pip
# can lose vendored urllib3 files while replacing the external urllib3 package,
# so optional packages are installed with uv after repairing pip.
"${ISAACLAB_PATH}/isaaclab.sh" -i core
ISAAC_PYTHON=/isaac-sim/kit/python/bin/python3
uv pip install --python "${ISAAC_PYTHON}" --force-reinstall pip==26.2.1
uv pip install --python "${ISAAC_PYTHON}" --constraint /tmp/isaaclab-install-constraints.txt \
    --no-deps \
    --editable "${ISAACLAB_PATH}/source/isaaclab_teleop" \
    --editable "${ISAACLAB_PATH}/source/isaaclab_mimic"
uv pip install --python "${ISAAC_PYTHON}" --constraint /tmp/isaaclab-install-constraints.txt \
    "isaacteleop[retargeters,ui,cloudxr]~=1.4.145" \
    "dex-retargeting==0.5.0" \
    "ipywidgets>=8.1.5" \
    "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0"
rm /tmp/isaaclab-install-constraints.txt
