#!/bin/bash
# Install the lightweight GR00T/OpenPI clients and their dependencies.

set -euo pipefail

# Install message serialization and transport libraries for communicating with policy servers.
/isaac-sim/python.sh -m pip install msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1
# Make the GR00T checkout importable without installing its full training/inference dependencies.
# Its metadata requires Python 3.10; the client here uses Isaac Sim's Python instead.
/isaac-sim/python.sh -m pip install --no-deps --ignore-requires-python -e "${WORKDIR}/submodules/Isaac-GR00T/"
# GR00T's video utilities import PyAV.
/isaac-sim/python.sh -m pip install av
# Read the pinned revision, removing whitespace, and install only OpenPI's client subpackage.
# Policy inference servers run separately from this Arena image.
OPENPI_COMMIT=$(tr -d '[:space:]' < /tmp/openpi_commit)
/isaac-sim/python.sh -m pip install --no-cache-dir \
    "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@${OPENPI_COMMIT}#subdirectory=packages/openpi-client"
