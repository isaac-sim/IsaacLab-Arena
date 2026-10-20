#!/bin/bash
# Install the lightweight GR00T/OpenPI clients and their dependencies.

set -euo pipefail

# Install message serialization and transport libraries for communicating with policy servers.
# Dependencies shared with Lab and Arena stay at the versions selected by their locks.
uv pip install --no-deps \
    av dm-tree msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1 wrapt
# Make the GR00T checkout importable without installing its full training/inference dependencies.
# Its metadata requires Python 3.10; the client here uses Isaac Sim's Python instead.
"${VIRTUAL_ENV}/bin/python" -m pip install \
    --no-deps --ignore-requires-python --editable "${WORKDIR}/submodules/Isaac-GR00T/"
# Read the pinned revision, removing whitespace, and install only OpenPI's client subpackage.
# Policy inference servers run separately from this Arena image. Its conservative
# NumPy upper bound is intentionally not resolved; Arena's lock carries the tested version.
OPENPI_COMMIT=$(tr -d '[:space:]' < /tmp/openpi_commit)
uv pip install --no-cache --no-deps \
    "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@${OPENPI_COMMIT}#subdirectory=packages/openpi-client"
