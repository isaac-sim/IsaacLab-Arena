#!/bin/bash
# Install the CUDA toolkit and build the cuRobo Python wheel.
set -euo pipefail

# Install the compiler toolkit in this builder stage and make its tools and libraries available.
bash /tmp/install_cuda.sh
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# These packages provide version detection, wheel packaging, and the build runner.
/isaac-sim/python.sh -m pip install setuptools_scm wheel ninja

# Build the pinned cuRobo revision into /wheels for the runtime stage.
# Skip dependency wheels and build in this environment instead of an isolated Python environment.
mkdir -p /wheels
/isaac-sim/python.sh -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels \
    "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@${CUROBO_COMMIT}"

# The Python block below reads the finished wheel's declared dependencies and writes
# a requirements file. The runtime stage installs those separately with version constraints
# that preserve Isaac Sim's existing packages.
/isaac-sim/python.sh - <<'PYTHON'
import email
from pathlib import Path
import zipfile

wheels = list(Path('/wheels').glob('*.whl'))
assert len(wheels) == 1, wheels
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    metadata_path = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
    # Wheel metadata uses email-style headers; the standard-library parser reads its dependency fields.
    metadata = email.message_from_bytes(archive.read(metadata_path))
    requirements = metadata.get_all('Requires-Dist', [])
Path('/wheels/runtime-requirements.txt').write_text('\n'.join(requirements) + '\n')
PYTHON
