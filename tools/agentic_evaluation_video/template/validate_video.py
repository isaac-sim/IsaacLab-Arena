# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check the current movie export and extract frames for visual review."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from edit_video import font, reference_bounds
from narration import rollout_length, rollout_rate, script_text, timeline, visual_time, words
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--media-only",
    action="store_true",
    help="Check an edit of existing footage without asserting fresh source provenance",
)
args = parser.parse_args()
results = {"source_provenance_checked": not args.media_only}
project = json.loads((root / "project.json").read_text())
capture_manifest = root / "capture_sources.json"
if not args.media_only:
    assert capture_manifest.exists(), "Record all capture stages first, or explicitly select --media-only"
    runtime = json.loads((root / "runtime.json").read_text())
    provenance = json.loads((root / runtime["source_provenance"]).read_text())
    for name, expected_hash in provenance["definition_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash, name
    captures = json.loads(capture_manifest.read_text())
    assert set(captures) == set(project["stages"])
    for stage, directory in captures.items():
        records = json.loads((root / directory / "captures.json").read_text())
        assert len(records) == project["capture_counts"][project["stages"].index(stage)]
        for record in records:
            frames = root / directory / f"layout_{record['layout']:02d}"
            assert len(record["trajectory"]) == len(list(frames.glob("frame_*.jpg"))) == 34
    experiment = json.loads((root / runtime["evaluation_directory"] / "arena_experiment_result.json").read_text())
    assert experiment["git_stamp"] == {"git_sha": provenance["commit"], "git_branch": provenance["branch"]}
    assert all(run["status"] == "completed" for run in experiment["runs"].values())
    results["source_verification"] = dict(
        commit=provenance["commit"],
        fresh_capture_stages=list(captures),
        evaluation_directory=runtime["evaluation_directory"],
    )
rollouts = json.loads((root / "rollouts.json").read_text())
assert len(rollouts) == 3
if project["require_front_left"]:
    assert sum(e.get("robot_side") == "left" for e in rollouts) == 1
assert len({(e["source"], e["episode"]) for e in rollouts}) == 3
if project["require_front_left"]:
    island_bounds = json.loads((root / "assets/rendered/sources.json").read_text())["island"]["bounds"]
for entry in rollouts:
    episodes = [json.loads(line) for line in (root / entry["results"]).read_text().splitlines()]
    episode = next(e for e in episodes if e["episode_in_env"] == entry["episode"])
    assert episode["success"] == entry["success"]
    if project["require_front_left"] and entry.get("robot_side") == "left":
        pose = episode["placement"]["poses"]["robot"]["position_xyz"]
        assert pose[0] > 0.2 and pose[1] > island_bounds[1][1], pose
        expected = json.loads((root / "front_left_sample.jsonl").read_text())["placement"]["poses"]
        for key, values in expected.items():
            for field in ["position_xyz", "rotation_xyzw"]:
                actual = episode["placement"]["poses"][key][field]
                assert max(abs(a - b) for a, b in zip(actual, values[field])) < 2e-5, (key, field)
        assert entry["robot_edge"] == "front"
evaluation = json.loads((root / "evaluation_summary.json").read_text())
timing = timeline()
assert timing == json.loads((root / "narration_timing.json").read_text())
assert timing["word_count"] == len(
    words(" ".join(line for line in script_text().splitlines() if not line.startswith("#")))
)
generation = json.loads((root / "audio/fixed_speed/generation.json").read_text())
assert generation["tempo_adjustment"] is False and timing["audio_time_stretched"] is False
assert {p["speed"] for p in generation["paragraphs"]} == {1.0}
for source, scheduled in zip(generation["paragraphs"], timing["paragraphs"]):
    assert abs(source["duration"] - (scheduled["end"] - scheduled["start"])) < 1 / 24000
previous_end = 0
for cue in timing["cues"]:
    assert previous_end <= cue["start"] < cue["end"] <= timing["duration"]
    if timing["audio_included"]:
        assert len(cue["word_starts"]) == len(words(cue["text"]))
        assert cue["word_starts"] == sorted(cue["word_starts"])
        assert cue["start"] <= cue["word_starts"][0] < cue["word_starts"][-1] < cue["end"]
    else:
        assert abs(len(words(cue["text"])) * 60 / (cue["end"] - cue["start"]) - 140) < 1e-6
    assert font(32, bold=False).getlength(cue["text"]) < 1810
    previous_end = cue["end"]
# Check the authoring moments line up with the corresponding narration phrases.
for fragment, expected_start in [
    ("kitchen environment,", 2),
    ("bowl on the countertop.", 9),
    ("adds a robot,", 18),
    ("remaining assets:", 27),
    ("composite task combines", 40),
    ("two pick-and-place tasks,", 41),
    ("Pi0.5 policy.", 47),
    ("launches the evaluation", 49.7),
]:
    cue = next(c for c in timing["cues"] if fragment in c["text"])
    assert abs(visual_time(cue["start"], timing["cues"]) - expected_start) < 1e-6
assert len((root / "subtitles/narration.srt").read_text().split(" --> ")) - 1 == len(timing["cues"])
assert not (root / "kitchenbench_split_bowls_authoring.srt").exists()
authoring_duration = timing["authoring_duration"]
expected_light_frames = round(timing["duration"] * 30)
for variant, filename in [("light", "kitchenbench_split_bowls_authoring.mp4")]:
    movie = root / filename
    metadata = json.loads(
        subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,duration,nb_read_frames",
            "-of",
            "json",
            str(movie),
        ])
    )["streams"][0]
    assert (metadata["width"], metadata["height"], metadata["r_frame_rate"]) == (1920, 1080, "30/1")
    assert int(metadata["nb_read_frames"]) == expected_light_frames
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(movie), "-f", "null", "-"], check=True)
    metadata.update(sha256=hashlib.sha256(movie.read_bytes()).hexdigest(), full_decode_passed=True)
    results[variant] = metadata
if timing["audio_included"]:
    audio_info = json.loads(
        subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(root / "kitchenbench_split_bowls_authoring.mp4"),
        ])
    )["streams"][0]
    assert audio_info["codec_name"] == "aac"
    assert int(audio_info["sample_rate"]) == 48000 and audio_info["channels"] == 1
    assert abs(float(audio_info["duration"]) - timing["duration"]) < 0.05
    results["audio"] = audio_info
review = root / "encoded_review"
review.mkdir(exist_ok=True)


def cue_midpoint(fragment):
    cue = next(c for c in timing["cues"] if fragment in c["text"])
    return round((cue["start"] + cue["end"]) / 2, 2)


times = [
    2,
    cue_midpoint("the scene, the task"),
    cue_midpoint("At runtime,"),
    cue_midpoint("two pick-and-place tasks"),
    cue_midpoint("Pi0.5 policy."),
    round(authoring_duration - 0.6, 2),
]
clip_start = authoring_duration
for entry in rollouts:
    times.append(round(clip_start + entry["frame_count"] / 30 / rollout_rate(timing) + 1.0, 2))
    clip_start += rollout_length(entry, timing)
for t in times:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            str(t),
            "-i",
            str(root / "kitchenbench_split_bowls_authoring.mp4"),
            "-frames:v",
            "1",
            str(review / f"{t:04.1f}.jpg"),
        ],
        check=True,
    )
# Verify captions survive encoding in every review frame that has an active cue.
for t in times:
    if any(c["start"] <= t < c["end"] for c in timing["cues"]):
        band = Image.open(review / f"{t:04.1f}.jpg").convert("L").crop((55, 1005, 1865, 1060))
        assert sum(count for value, count in enumerate(band.histogram()) if value < 120) > 500
# Check that completed episodes retain their recorded outcome, while future slots stay neutral.
for completed, t in enumerate(times[6:], 1):
    encoded = Image.open(review / f"{t:04.1f}.jpg").convert("RGB")
    for slot, xy in enumerate([(75, 945), (688, 945), (1301, 945)]):
        expected = (
            ((118, 185, 0) if rollouts[slot]["success"] else (196, 61, 54)) if slot < completed else (238, 238, 238)
        )
        assert max(abs(a - b) for a, b in zip(encoded.getpixel(xy), expected)) < 20, (
            completed,
            slot,
            encoded.getpixel(xy),
            expected,
        )
# Check the three introductory pulses in the encoded movie, including their order.
highlight_times = []
for target in ["scene", "task", "policy"]:
    event = next(e for e in timing["references"] if e["phrase"] == target)
    t = (event["start"] + event["end"]) / 2
    highlight_times.append(t)
    path = review / f"highlight_{target}.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            str(t),
            "-i",
            str(root / "kitchenbench_split_bowls_authoring.mp4"),
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
    )
    encoded = Image.open(path).convert("RGB")
    for box in ["scene", "task", "policy"]:
        x1, y1, x2, y2 = reference_bounds(box)
        pixel = encoded.getpixel((x1 + 40, y1 + 3))
        if box == target:
            assert max(abs(a - b) for a, b in zip(pixel, (118, 185, 0))) < 25, (target, pixel)
        else:
            assert pixel[2] > 100, (target, box, pixel)
assert highlight_times == sorted(highlight_times)
# The opening retains the three inner panels, and the quoted prompt pulses independently.
opening = Image.open(review / f"{times[0]:04.1f}.jpg").convert("RGB")
for xy in [(700, 450), (700, 700), (700, 860)]:
    assert max(abs(c - 238) for c in opening.getpixel(xy)) < 10
prompt_event = next(e for e in timing["references"] if e["phrase"] == "Design")
prompt_time = (prompt_event["start"] + prompt_event["end"]) / 2
prompt_path = review / "highlight_prompt.jpg"
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        str(prompt_time),
        "-i",
        str(root / "kitchenbench_split_bowls_authoring.mp4"),
        "-frames:v",
        "1",
        str(prompt_path),
    ],
    check=True,
)
x1, y1, _, _ = reference_bounds("prompt")
pixel = Image.open(prompt_path).convert("RGB").getpixel((x1 + 40, y1 + 3))
assert max(abs(a - b) for a, b in zip(pixel, (118, 185, 0))) < 25, pixel
streams = json.loads(
    subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(root / "kitchenbench_split_bowls_authoring.mp4"),
    ])
)["streams"]
assert [s["codec_type"] for s in streams] == (["video", "audio"] if timing["audio_included"] else ["video"])
sheet = Image.new("RGB", (1440, 882), "#ffffff")
for i, t in enumerate(times):
    x, y = (i % 3) * 480, (i // 3) * 294
    sheet.paste(Image.open(review / f"{t:04.1f}.jpg").resize((480, 270)), (x, y))
    ImageDraw.Draw(sheet).text((x + 8, y + 274), str(t) + " s", fill="#20251c")
sheet.save(root / "review_contact_sheet.jpg", quality=94)
Image.open(review / f"{times[5]:04.1f}.jpg").save(root / "poster.jpg", quality=94)
results.update(
    selected_rollouts=len(rollouts),
    selected_successful_rollouts=sum(e["success"] for e in rollouts),
    filmed_attempts=evaluation["episodes"],
    filmed_successes=evaluation["successes"],
    both_final_placements_required=True,
    source_episodes=[{"results": e["results"], "episode": e["episode"]} for e in rollouts],
    playback_speed=rollouts[0]["speed"] * rollout_rate(timing),
    left_side_rollouts=sum(e.get("robot_side") == "left" for e in rollouts),
    outcome_boxes_verified=True,
    caption_count=len(timing["cues"]),
    caption_words=timing["word_count"],
    caption_wpm=timing["words_per_minute"],
    caption_timing_and_width_verified=True,
    introductory_reference_pulses_verified=True,
    single_caption_source_verified=True,
    opening_panels_verified=True,
    prompt_highlight_verified=True,
    fixed_voice_speed_verified=True,
    front_left_sample_replay_verified=project["require_front_left"],
    audio_included=timing["audio_included"],
    review_frames_s=times,
)
(root / "video_validation.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
