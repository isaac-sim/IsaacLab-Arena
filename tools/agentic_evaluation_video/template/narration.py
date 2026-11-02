# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Caption cues and authoring animation timing for the narration script."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORDS_PER_MINUTE = 140
# Each phrase maps to the original animation clock; repeated values hold a view.
# None selects evaluation footage, whose timing is preserved independently.
PARAGRAPHS = [
    (
        "Title screen",
        [
            ("Here, we show how an agent can author", 0, 0),
            ("a robotics evaluation using Isaac Lab - Arena.", 0, 0),
        ],
    ),
    (
        "Composition",
        [
            ("For this example we use the prompt:", 0, 0),
            ("“Design an evaluation environment in a kitchen", 0, 0),
            ("where two pieces of fruit must be placed into two bowls.”", 0, 0),
        ],
    ),
    (
        "Composition",
        [
            ("An Isaac Lab - Arena evaluation environment consists of three components:", 0, 0),
            ("the scene, the task, and the policy.", 0, 1.9),
        ],
    ),
    (
        "Scene",
        [
            ("In Arena, a scene is composed of USD assets", 1.9, 1.9),
            ("connected through spatial relations.", 1.9, 1.9),
        ],
    ),
    (
        "Scene",
        [
            ("The agent starts by adding a kitchen environment,", 2, 8.9),
            ("followed by a bowl on the countertop.", 9, 12.49),
        ],
    ),
    (
        "Scene",
        [
            ("At runtime, the spatial graph generates randomized layouts,", 12.5, 16),
            ("several of which are shown on the right.", 16, 17.9),
        ],
    ),
    (
        "Scene",
        [
            ("The agent then adds a robot,", 18, 26.9),
            ("followed by the remaining assets: the fruit and a second bowl.", 27, 38.9),
        ],
    ),
    (
        "Task",
        [
            ("Next, the agent specifies the goal using Arena's task system.", 39, 39.95),
        ],
    ),
    (
        "Task",
        [
            ("A composite task combines the two placement objectives.", 40, 40.95),
        ],
    ),
    (
        "Task",
        [
            ("The agent creates two pick-and-place tasks,", 41, 43),
            ("each referencing the appropriate assets in the scene graph.", 43, 43.9),
        ],
    ),
    (
        "Policy",
        [
            ("In this video, we evaluate the Pi0.5 policy.", 47, 49.6),
        ],
    ),
    (
        "Evaluation",
        [
            ("Finally, the agent launches the evaluation in Isaac Lab.", 49.7, 53),
        ],
    ),
    (
        "Evaluation",
        [
            ("Arena samples valid layouts and evaluates", None, None),
            ("whether the policy completes both placements.", None, None),
        ],
    ),
    (
        "Evaluation",
        [
            ("In this example we show the DROID-finetuned pi0.5 model", None, None),
            ("deployed zero-shot to complete a composite pick and place task.", None, None),
        ],
    ),
    (
        "Evaluation",
        [
            ("The policy must deposit the close fruit in the close bowl,", None, None),
            ("and the far fruit in the far bowl.", None, None),
        ],
    ),
    (
        "Evaluation",
        [
            ("Each rollout records a success or failure.", None, None),
        ],
    ),
]


def words(text):
    return re.findall(r"\b[\w]+(?:[’'.-][\w]+)*\b", text)


# Phrase matching keeps emphasis aligned to the estimated spoken-word timings.
REFERENCES = [
    ("Here, we show", "agent", ["agent"]),
    ("a robotics evaluation", "Isaac Lab", ["arena"]),
    ("For this example", "prompt", ["prompt"]),
    ("“Design an evaluation", "Design", ["prompt"]),
    ("An Isaac Lab - Arena evaluation", "Isaac Lab", ["arena"]),
    ("the scene, the task", "scene", ["scene"]),
    ("the scene, the task", "task", ["task"]),
    ("the scene, the task", "policy", ["policy"]),
    ("In Arena, a scene", "scene", ["scene"]),
    ("connected through", "spatial relations", ["scene"]),
    ("The agent starts", "kitchen environment", ["kitchen"]),
    ("followed by a bowl", "bowl", ["near"]),
    ("At runtime,", "spatial graph", ["scene"]),
    ("several of which", "on the right", ["simulator"]),
    ("The agent then", "robot", ["robot"]),
    ("followed by the remaining", "fruit", ["banana", "orange"]),
    ("followed by the remaining", "second bowl", ["far"]),
    ("Next, the agent", "goal", ["task"]),
    ("A composite task", "composite task", ["composite"]),
    ("The agent creates", "two pick-and-place tasks", ["pick_banana", "pick_orange"]),
    ("each referencing", "assets", ["banana", "orange", "near", "far"]),
    ("In this video,", "Pi0.5", ["policy"]),
    ("Finally, the agent", "Isaac Lab", ["simulator"]),
    ("Arena samples valid", "layouts", ["rollout_video"]),
    ("whether the policy", "policy", ["rollout_policy"]),
    ("whether the policy", "both placements", ["rollout_task"]),
    ("In this example we show", "DROID-finetuned", ["rollout_policy"]),
    ("deployed zero-shot", "composite", ["rollout_task"]),
    ("The policy must deposit", "close fruit", ["rollout_banana"]),
    ("and the far fruit", "far fruit", ["rollout_orange"]),
    ("Each rollout records", "success or failure", ["results"]),
]


def reference_events(cues):
    events = []
    for fragment, phrase, targets in REFERENCES:
        cue = next(c for c in cues if fragment in c["text"])
        tokens = words(cue["text"])
        mentioned = words(phrase)
        index = next(i for i in range(len(tokens)) if tokens[i : i + len(mentioned)] == mentioned)
        start = cue["word_starts"][index] if "word_starts" in cue else cue["start"] + index * 60 / WORDS_PER_MINUTE
        # Wait for a new asset/card to be visible before outlining it.
        minimum = {"near": 10.8, "pick_orange": 41.6}
        for target in targets:
            if target in minimum and cue["visual_start"] is not None:
                fraction = (minimum[target] - cue["visual_start"]) / (cue["visual_end"] - cue["visual_start"])
                start = max(start, cue["start"] + fraction * (cue["end"] - cue["start"]))
        events.append(dict(start=start, end=start + 0.72, targets=targets, phrase=phrase))
    return events


def script_text():
    lines = []
    previous = None
    for section, phrases in PARAGRAPHS:
        if section != previous:
            lines.append(f"### {section}")
            previous = section
        lines.append(" ".join(phrase[0] for phrase in phrases))
    return "\n\n".join(lines) + "\n"


def timeline():
    assert (
        ROOT / "VOICE_OVER.md"
    ).read_text().strip() == script_text().strip(), "Update narration phrase timings after changing the script."
    audio_timing = ROOT / "narration_audio_timing.json"
    if audio_timing.exists():
        data = json.loads(audio_timing.read_text())
        assert (
            data["script_sha256"] == hashlib.sha256((ROOT / "VOICE_OVER.md").read_bytes()).hexdigest()
        ), "Regenerate the voice-over after changing the script."
        assert (ROOT / data["audio_file"]).exists()
        data["references"] = reference_events(data["cues"])
        return data
    rollouts = json.loads((ROOT / "rollouts.json").read_text())
    cues = []
    t = 0.6
    authoring_end = None
    for paragraph, (section, phrases) in enumerate(PARAGRAPHS):
        if paragraph == len(PARAGRAPHS) - 1:
            # Let the viewer watch the remaining policy footage, then narrate the outcomes.
            t = authoring_end + sum(e["frame_count"] / 30 + 1.8 for e in rollouts) - 1.3
        for phrase, start, end in phrases:
            if start is None and authoring_end is None:
                authoring_end = t
            duration = len(words(phrase)) * 60 / WORDS_PER_MINUTE
            cues.append(
                dict(start=t, end=t + duration, text=phrase, section=section, visual_start=start, visual_end=end)
            )
            t += duration
        t += 0.3
    return dict(
        words_per_minute=WORDS_PER_MINUTE,
        word_count=sum(len(words(c["text"])) for c in cues),
        authoring_duration=authoring_end,
        duration=t + 0.7,
        cues=cues,
        audio_included=False,
        references=reference_events(cues),
    )


def rollout_rate(data):
    return data.get("rollout_frame_rate_scale", 1.0)


def rollout_length(entry, data):
    return entry["frame_count"] / 30 / rollout_rate(data) + 1.8


def visual_time(t, cues):
    previous = 0
    for cue in cues:
        if cue["visual_start"] is None:
            break
        if t < cue["start"]:
            return previous
        if t < cue["end"]:
            fraction = (t - cue["start"]) / (cue["end"] - cue["start"])
            return cue["visual_start"] + fraction * (cue["visual_end"] - cue["visual_start"])
        previous = cue["visual_end"]
    return previous


def export_timing(data):
    (ROOT / "narration_timing.json").write_text(json.dumps(data, indent=2) + "\n")

    def stamp(t):
        milliseconds = round(t * 1000)
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    subtitles = []
    for i, cue in enumerate(data["cues"], 1):
        subtitles.append(f"{i}\n{stamp(cue['start'])} --> {stamp(cue['end'])}\n{cue['text']}\n")
    # Keep the editable subtitle track away from the movie so players do not
    # automatically overlay it on the captions already rendered into the video.
    folder = ROOT / "subtitles"
    folder.mkdir(exist_ok=True)
    (folder / "narration.srt").write_text("\n".join(subtitles))


if __name__ == "__main__":
    (ROOT / "VOICE_OVER.md").write_text(script_text())
    data = timeline()
    export_timing(data)
    print(json.dumps({k: v for k, v in data.items() if k != "cues"}, indent=2))
