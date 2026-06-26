# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tile per-episode external_camera_rgb videos into one grid video with progress overlays.

For each finished episode in a run dir (episode_results_rank*.jsonl + matching
robot-cam-env<N>-external_camera_rgb-episode-<E>.mp4), draw a subtask legend whose markers
flip red -> green at the frame where the progress tracker recorded that predicate complete,
then tile all episodes into a single mp4.

Frame/step alignment: the recorder drops each episode's first (reset) frame, so the video has
``episode_length - 1`` frames and frame index ``f`` corresponds to env step ``s = f + 1``. A
predicate completing at step ``c`` therefore flips at frame index ``c - 1`` (clamped to the last
frame, so a terminal-step completion shows on the final frame).

Usage: python tile_episodes_with_progress.py <RUN_DIR> [-o OUT.mp4] [--camera external_camera_rgb]

/isaac-sim/python.sh progress_tracking_visualization/tile_episodes_with_progress.py \
  <PATH_TO_FRAMES_DIRECTORY> \
  --hold_secs 2 \
  -o <PATH_TO_OUTPUT_DIR/FILENAME.mp4>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

from isaaclab_arena.video.camera_observation_video_recorder import parse_episode_video_filename

CELL_W, CELL_H = 640, 360
RED = (60, 60, 230)  # BGR
GREEN = (60, 200, 60)
WHITE = (245, 245, 245)
DIM = (180, 180, 180)


def short_name(predicate_name: str) -> str:
    """``object_lifted(object_name=...)`` -> ``object_lifted``."""
    return predicate_name.split("(", 1)[0]


def load_episodes(run_dir: str, camera: str):
    """Return a list of per-episode dicts joined to their video, sorted by (env_id, episode_in_env)."""
    records = []
    for jf in sorted(glob.glob(os.path.join(run_dir, "episode_results_rank*.jsonl"))):
        with open(jf) as f:
            records.extend(json.loads(line) for line in f if line.strip())

    # Discover the global subtask label map {predicate_index: short_name} from any record's events.
    labels: dict[int, str] = {}
    for r in records:
        for ev in r.get("progress", {}).get("events", []):
            labels.setdefault(ev["predicate_index"], short_name(ev["predicate_name"]))
    num_subtasks = (max(labels) + 1) if labels else 0
    labels = {i: labels.get(i, f"subtask {i}") for i in range(num_subtasks)}

    episodes = []
    for r in records:
        env_id, ep = r["env_id"], r["episode_in_env"]
        path = os.path.join(run_dir, f"robot-cam-env{env_id}-{camera}-episode-{ep}.mp4")
        if not os.path.exists(path):
            print(f"  WARN no video for env{env_id} ep{ep} ({camera}) -> skipped")
            continue
        cap = cv2.VideoCapture(path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # completion frame index per subtask: step c -> frame c-1, clamped into [0, n_frames-1]
        completion = {}
        for ev in r.get("progress", {}).get("events", []):
            completion[ev["predicate_index"]] = min(max(ev["step"] - 1, 0), n_frames - 1)
        episodes.append({
            "env_id": env_id, "ep": ep, "success": r["success"],
            "cap": cap, "n_frames": n_frames, "last": None,
            "completion": completion, "ep_len": r["episode_length"],
        })
    episodes.sort(key=lambda e: (e["env_id"], e["ep"]))
    return episodes, labels


def draw_overlay(frame, epi, labels, t):
    """Draw the title + subtask legend on a CELL_W x CELL_H frame for output-frame index t."""
    ended = t >= epi["n_frames"] - 1
    step_shown = min(t + 1, epi["ep_len"])

    # Title bar.
    cv2.rectangle(frame, (0, 0), (CELL_W, 26), (30, 30, 30), -1)
    cv2.putText(frame, f"env{epi['env_id']} ep{epi['ep']}   step {step_shown}/{epi['ep_len']}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if ended:
        tag, col = ("SUCCESS", GREEN) if epi["success"] else ("FAIL", RED)
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, tag, (CELL_W - tw - 8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    # Legend box (bottom-left), one row per subtask.
    n = len(labels)
    row_h = 22
    box_h = n * row_h + 10
    y0 = CELL_H - box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (CELL_W, CELL_H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i in range(n):
        cy = y0 + 8 + i * row_h + 8
        done = i in epi["completion"] and t >= epi["completion"][i]
        cv2.circle(frame, (16, cy), 7, GREEN if done else RED, -1)
        cv2.circle(frame, (16, cy), 7, (240, 240, 240), 1, cv2.LINE_AA)
        txt = f"{i + 1}. {labels[i]}"
        cv2.putText(frame, txt, (32, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    WHITE if done else DIM, 1, cv2.LINE_AA)

    # Once the episode has ended, frame the whole tile so the outcome is unmistakable.
    if ended:
        cv2.rectangle(frame, (2, 2), (CELL_W - 3, CELL_H - 3), GREEN if epi["success"] else RED, 5)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--camera", default="external_camera_rgb")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--hold_secs", type=float, default=0.0,
                    help="Freeze on the final composed frame for this many seconds at the end.")
    args = ap.parse_args()

    episodes, labels = load_episodes(args.run_dir, args.camera)
    assert episodes, f"No episodes with {args.camera} videos found in {args.run_dir}"
    print(f"{len(episodes)} episodes, subtasks: {labels}")

    fps = args.fps or 50
    total_frames = max(e["n_frames"] for e in episodes)
    n = len(episodes)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    out_w, out_h = cols * CELL_W, rows * CELL_H
    out_path = args.out or os.path.join(args.run_dir, "episode_tiles_progress.mp4")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    print(f"grid {cols}x{rows} -> {out_w}x{out_h}, {total_frames} frames @ {fps}fps -> {out_path}")

    for t in range(total_frames):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for idx, epi in enumerate(episodes):
            if t < epi["n_frames"]:
                ok, fr = epi["cap"].read()
                if ok:
                    epi["last"] = fr
            fr = epi["last"]
            if fr is None:
                continue
            cell = cv2.resize(fr, (CELL_W, CELL_H))
            cell = draw_overlay(cell, epi, labels, t)
            r, c = divmod(idx, cols)
            canvas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cell
        writer.write(canvas)

    # Freeze on the final composed frame for --hold_secs seconds.
    hold_frames = int(round(args.hold_secs * fps))
    for _ in range(hold_frames):
        writer.write(canvas)
    if hold_frames:
        print(f"held final frame for {args.hold_secs}s ({hold_frames} frames)")

    writer.release()
    for e in episodes:
        e["cap"].release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
