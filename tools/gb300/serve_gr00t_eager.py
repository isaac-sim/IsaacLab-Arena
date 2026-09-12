# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch GR00T with PyTorch eager attention for the temporary GB300 setup."""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path

DEFAULT_GR00T_ROOT = Path("/workspaces/isaaclab_arena/submodules/Isaac-GR00T")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-root", type=Path, default=DEFAULT_GR00T_ROOT)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.6-DROID")
    parser.add_argument("--embodiment-tag", default="OXE_DROID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gr00t_root = args.gr00t_root.resolve()
    site_packages = args.site_packages or gr00t_root / ".venv-sbsa/lib/python3.12/site-packages"
    server_script = gr00t_root / "gr00t/eval/run_gr00t_server.py"
    assert site_packages.is_dir(), f"GR00T dependency directory not found: {site_packages}"
    assert server_script.is_file(), f"GR00T server script not found: {server_script}"
    assert 1 <= args.port <= 65535, f"Invalid port: {args.port}"

    sys.path.insert(0, str(site_packages))
    sys.path.insert(0, str(gr00t_root))

    # Load both modules before GR00T so torchvision operators register against this torch runtime.
    importlib.import_module("torch")
    importlib.import_module("torchvision")
    gr00t_policy = importlib.import_module("gr00t.policy.gr00t_policy")
    original_init = gr00t_policy.Gr00tPolicy.__init__

    def eager_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        changed = 0
        for module in self.model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
                changed += 1
        print(f"Switched {changed} loaded module configuration(s) to eager attention", flush=True)

    gr00t_policy.Gr00tPolicy.__init__ = eager_init
    sys.argv = [
        str(server_script),
        "--model-path",
        args.model_path,
        "--embodiment-tag",
        args.embodiment_tag,
        "--device",
        args.device,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    runpy.run_path(str(server_script), run_name="__main__")


if __name__ == "__main__":
    main()
