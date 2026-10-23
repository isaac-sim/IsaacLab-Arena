#!/bin/bash
# Install the CUDA toolkit and build the cuRobo Python wheel.
set -euo pipefail

# Install the compiler toolkit in this builder stage and make its tools and libraries available.
bash /tmp/install_cuda.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# These packages provide version detection, wheel packaging, and the build runner.
uv pip install setuptools_scm wheel ninja

# Fetch the pinned cuRobo revision so its Warp compatibility can be patched before wheel creation.
export CUROBO_SOURCE_DIR=/tmp/curobo
git init "${CUROBO_SOURCE_DIR}"
git -C "${CUROBO_SOURCE_DIR}" remote add origin https://github.com/NVlabs/curobo.git
git -C "${CUROBO_SOURCE_DIR}" fetch --depth=1 origin "${CUROBO_COMMIT}"
git -C "${CUROBO_SOURCE_DIR}" checkout --detach FETCH_HEAD

# cuRobo 0.7.x uses the removed Warp 1.17 wp.torch namespace. The interop
# functions remain available at Warp's package root, so update the one stale call.
"${VIRTUAL_ENV}/bin/python" - <<'PYTHON'
import os
from pathlib import Path

path = Path(os.environ["CUROBO_SOURCE_DIR"]) / "src/curobo/geom/sdf/world_mesh.py"
old = "wp.torch.device_from_torch(self.tensor_args.device)"
new = "wp.device_from_torch(self.tensor_args.device)"
contents = path.read_text()
assert contents.count(old) == 1, f"Expected one cuRobo Warp interop call in {path}"
path.write_text(contents.replace(old, new))
PYTHON

# Build the patched revision into /wheels for the runtime stage. Skip dependency
# wheels and build in this environment instead of an isolated Python environment.
mkdir -p /wheels
"${VIRTUAL_ENV}/bin/python" -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels \
    "${CUROBO_SOURCE_DIR}"

# The Python block below reads the finished wheel's declared dependencies and writes
# a requirements file. The runtime stage installs those separately with version constraints
# that preserve Isaac Sim's existing packages.
"${VIRTUAL_ENV}/bin/python" - <<'PYTHON'
import email
import hashlib
import json
import os
from pathlib import Path
import zipfile

import torch

wheels = list(Path('/wheels').glob('*.whl'))
assert len(wheels) == 1, wheels
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    metadata_path = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
    # Wheel metadata uses email-style headers; the standard-library parser reads its dependency fields.
    metadata = email.message_from_bytes(archive.read(metadata_path))
    requirements = metadata.get_all('Requires-Dist', [])
Path('/wheels/runtime-requirements.txt').write_text('\n'.join(requirements) + '\n')
# Human-readable provenance for diagnosing installed-wheel revision/ABI mismatches.
Path('/wheels/build.json').write_text(json.dumps({
    'commit': os.environ['CUROBO_COMMIT'],
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'architectures': os.environ['TORCH_CUDA_ARCH_LIST'],
    'wheel': wheel.name,
    'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
}, indent=2) + '\n')
PYTHON
