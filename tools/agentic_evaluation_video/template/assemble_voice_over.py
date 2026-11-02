# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time the video to the natural Kokoro takes without changing their speaking rate."""
import hashlib
import json
import numpy as np
import subprocess
from pathlib import Path

import soundfile as sf
from narration import PARAGRAPHS, reference_events, rollout_length, script_text, words

ROOT = Path(__file__).resolve().parent


def main():
    script = (ROOT / "VOICE_OVER.md").read_text()
    assert script.strip() == script_text().strip(), "Update caption phrases to match VOICE_OVER.md."
    report = json.loads((ROOT / "audio/fixed_speed/generation.json").read_text())
    assert report["script_sha256"] == hashlib.sha256(script.encode()).hexdigest()
    assert report["tempo_adjustment"] is False
    assert len(report["paragraphs"]) == len(PARAGRAPHS)
    rollouts = json.loads((ROOT / "rollouts.json").read_text())
    evaluation_spoken = sum(
        p["duration"] + 0.3
        for p, (_, phrases) in zip(report["paragraphs"][:-1], PARAGRAPHS[:-1])
        if phrases[0][1] is None
    )
    media_duration = sum(e["frame_count"] / 30 for e in rollouts)
    holds = 1.8 * len(rollouts)
    # If needed, slow the evaluation footage to leave room for its narration.
    rate = min(1.0, media_duration / max(0.1, evaluation_spoken + 1.3 - holds))
    data = dict(
        words_per_minute=round(report["word_count"] * 60 / report["speech_duration"], 1),
        word_count=report["word_count"],
        audio_included=True,
        audio_file="voice_over.wav",
        script_sha256=report["script_sha256"],
        rollout_frame_rate_scale=rate,
        narration_speed=report["speed"],
        audio_time_stretched=False,
    )
    t = 0.6
    cues = []
    paragraphs = []
    for i, (take, (section, phrases)) in enumerate(zip(report["paragraphs"], PARAGRAPHS)):
        assert take["text"] == " ".join(p[0] for p in phrases)
        if phrases[0][1] is None and "authoring_duration" not in data:
            data["authoring_duration"] = t
        if i == len(PARAGRAPHS) - 1:
            t = max(t, data["authoring_duration"] + sum(rollout_length(e, data) for e in rollouts) - 1.3)
        start = t
        end = t + take["duration"]
        paragraph_words = take["words"]
        word_index = 0
        for text, visual_start, visual_end in phrases:
            count = len(words(text))
            selected = paragraph_words[word_index : word_index + count]
            cue_start = start + selected[0]["start"] if word_index else start
            word_index += count
            cue_end = start + paragraph_words[word_index]["start"] if word_index < len(paragraph_words) else end
            cues.append(
                dict(
                    start=cue_start,
                    end=cue_end,
                    text=text,
                    section=section,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    word_starts=[start + w["start"] for w in selected],
                )
            )
        paragraphs.append(dict(start=start, end=end, audio_file=take["audio_file"], number=take["number"]))
        t = end + 0.3
    data.update(duration=t + 0.7, cues=cues, paragraphs=paragraphs)
    data["references"] = reference_events(cues)
    rate_hz = report["sample_rate"]
    track = np.zeros(round(data["duration"] * rate_hz), dtype=np.float32)
    for p in paragraphs:
        audio, sr = sf.read(ROOT / p["audio_file"], dtype="float32")
        assert sr == rate_hz
        offset = round(p["start"] * rate_hz)
        track[offset : offset + len(audio)] = audio
    raw = ROOT / "audio/fixed_speed/voice_over_timed_raw.wav"
    sf.write(raw, track, rate_hz, subtype="FLOAT")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(raw),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s24le",
            str(ROOT / "voice_over.wav"),
        ],
        check=True,
    )
    (ROOT / "narration_audio_timing.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k not in ("cues", "references", "paragraphs")}, indent=2))


if __name__ == "__main__":
    main()
