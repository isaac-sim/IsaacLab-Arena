#!/bin/bash
# Build the cuRobo Python wheel.
set -euo pipefail

# cuRobo v2 compiles its CUDA kernels at runtime through cuda.core (NVRTC), so the wheel is pure Python and
# the build needs no CUDA toolkit. These packages provide version detection and wheel packaging.
uv pip install setuptools_scm wheel

# Build the pinned cuRobo revision into /wheels for the runtime stage.
# Skip dependency wheels and build in this environment instead of an isolated Python environment.
mkdir -p /wheels
"${VIRTUAL_ENV}/bin/python" -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels \
    "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@${CUROBO_COMMIT}"

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
# Keep the base dependencies plus the runtime-kernel extra matching torch's CUDA major (e.g. cu13), with its marker
# stripped; requirements gated on any other extra are dropped.
cuda_extra_marker = f'extra == "cu{torch.version.cuda.split(".")[0]}"'
runtime_requirements = [
    requirement.split(';')[0].strip() if cuda_extra_marker in requirement else requirement
    for requirement in requirements
    if 'extra ==' not in requirement or cuda_extra_marker in requirement
]
Path('/wheels/runtime-requirements.txt').write_text('\n'.join(runtime_requirements) + '\n')
# Human-readable provenance for diagnosing installed-wheel revision mismatches.
Path('/wheels/build.json').write_text(json.dumps({
    'commit': os.environ['CUROBO_COMMIT'],
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'wheel': wheel.name,
    'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
}, indent=2) + '\n')
PYTHON
