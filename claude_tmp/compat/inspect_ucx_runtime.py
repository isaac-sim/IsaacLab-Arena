from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser("inspect_ucx_runtime")
    parser.add_argument("--ucx-info-bin", type=str, default="ucx_info")
    parser.add_argument("--prepend-ld-library-path", type=str, default="")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.prepend_ld_library_path:
        env["LD_LIBRARY_PATH"] = args.prepend_ld_library_path + ":" + env.get("LD_LIBRARY_PATH", "")

    proc = subprocess.run(
        [args.ucx_info_bin, "-d"],
        env=env,
        capture_output=True,
        text=True,
    )

    lines = []
    for line in proc.stdout.splitlines():
        if any(
            token in line
            for token in [
                "Memory domain: mlx5",
                "Transport: rc",
                "Transport: dc",
                "Transport: ud",
                "Connection manager: rdmacm",
                "Connection manager: sockcm",
                "Device: mlx5",
            ]
        ):
            lines.append(line)

    print("RET", proc.returncode)
    if env.get("LD_LIBRARY_PATH"):
        print("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
    for line in lines:
        print(line)

    if proc.returncode != 0:
        sys.stdout.write(proc.stderr)
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
