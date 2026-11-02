# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the filmed evaluations and preserve all attempted outcomes."""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("evaluations", nargs="*", type=Path)
args = parser.parse_args()
records = []
for output in args.evaluations or [root / "evaluation"]:
    folder = str(output.resolve().relative_to(root))
    runs = json.loads((output / "arena_experiment_result.json").read_text())["runs"]
    assert (output / "index.html").is_file()
    for name, run in runs.items():
        assert run["status"] == "completed"
        canonical = [e for rebuild in run["rebuilds"] for e in rebuild["episodes"]]
        native = [
            json.loads(line) for line in (output / name / "episode_results_rebuild0.jsonl").read_text().splitlines()
        ]
        expected = len(canonical)
        assert expected > 0 and len(native) == expected
        assert len({(e["env_id"], e["episode_in_env"]) for e in native}) == expected
        for a, b in zip(canonical, native):
            for key in ("env_id", "episode_in_env", "success", "episode_length"):
                assert a[key] == b[key]
            if b["success"]:
                assert b["progress"]["all_complete"]
                assert all(c["is_complete"] for c in b["progress"]["criteria_by_name"].values())
            records.append({
                "evaluation": folder,
                "episode": b["episode_in_env"],
                "success": b["success"],
                "steps": b["episode_length"],
            })
summary = {
    "finalized": True,
    "episodes": len(records),
    "successes": sum(e["success"] for e in records),
    "require_both_at_end": True,
    "policy": "pi05",
    "endpoint": "127.0.0.1:8001",
    "results": records,
}
(root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
