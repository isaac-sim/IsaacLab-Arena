# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve a pinned GR00T N1.7 DROID checkpoint in its separate inference environment."""

import argparse
import contextlib
import random
import sys
from pathlib import Path

DEFAULT_MODEL = "nvidia/GR00T-N1.7-DROID"
DEFAULT_MODEL_REVISION = "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-root", type=Path, default=Path.home() / ".cache/arena-gr00t-n17-src")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision", help="Hugging Face revision; the default DROID model uses a pinned revision."
    )
    parser.add_argument("--embodiment-tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert 1 <= args.port <= 65535, "Invalid server port"
    assert (args.gr00t_root / "gr00t/model/gr00t_n1d7").is_dir(), "Use the separate Isaac-GR00T N1.7 checkout"
    sys.path.insert(0, str(args.gr00t_root.resolve()))

    import numpy as np
    import torch

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from huggingface_hub import snapshot_download

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_dir():
        revision = args.model_revision
        if revision is None and args.model_path == DEFAULT_MODEL:
            revision = DEFAULT_MODEL_REVISION
        model_path = Path(snapshot_download(args.model_path, revision=revision))

    print(f"GR00T N1.7 checkpoint: {model_path}; embodiment: {args.embodiment_tag}; seed: {args.seed}", flush=True)
    # Upstream Qwen3-VL falls back to PyTorch SDPA when FlashAttention is absent.
    # No torch.compile or N1.6 Eagle-specific monkey patches are needed.
    policy = Gr00tPolicy(args.embodiment_tag, str(model_path), device=args.device, strict=True)
    server = PolicyServer(policy, host=args.host, port=args.port)
    try:
        with contextlib.suppress(KeyboardInterrupt):
            server.run()
    finally:
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
