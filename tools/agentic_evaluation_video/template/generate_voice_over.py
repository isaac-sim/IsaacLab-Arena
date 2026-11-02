# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a Kokoro narration track and align captions to its word timestamps."""
import hashlib
import importlib.metadata
import json
import numpy as np
import re
import shutil
import subprocess
import sys
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

import espeakng_loader
import soundfile as sf
from kokoro import KPipeline
from misaki import espeak
from narration import words

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio/fixed_speed"
SAMPLE_RATE = 24000
VOICE = "af_heart"
MODEL = "hexgrad/Kokoro-82M"
SPEED = 1.0
PRONUNCIATIONS = {"pi0.5": "pie zero point five", "USD": "U S D", "DROID": "droid"}


def spoken_text(text):
    for original, spoken in PRONUNCIATIONS.items():
        text = re.sub(re.escape(original), spoken, text, flags=re.IGNORECASE)
    return (
        text.replace("Isaac Lab - Arena", "Isaac Lab Arena")
        .replace("pick-and-place", "pick and place")
        .replace("finetuned", "fine tuned")
        .replace("zero-shot", "zero shot")
        .replace("droid-", "droid ")
    )


def letters(text):
    return "".join(c for c in text.lower() if c.isalnum())


def word_timestamps(original, tokens, time_scale):
    """Map Kokoro tokens, including expanded pronunciations, back to script words."""
    character = 0
    spans = []
    token_text = ""
    for token in tokens:
        text = letters(token.text)
        if text:
            assert token.start_ts is not None and token.end_ts is not None, token.text
            spans.append((character, character + len(text), token.start_ts, token.end_ts))
            token_text += text
            character += len(text)
    expected = "".join(letters(spoken_text(w)) for w in words(original))
    assert token_text == expected, (token_text, expected)
    output = []
    character = 0
    for word in words(original):
        end = character + len(letters(spoken_text(word)))
        matches = [s for s in spans if s[0] < end and s[1] > character]
        assert matches, word
        output.append({"word": word, "start": matches[0][2] * time_scale, "end": matches[-1][3] * time_scale})
        character = end
    return output


def generate_narration():
    script = (ROOT / "VOICE_OVER.md").read_text()
    sections = []
    section = ""
    for block in script.strip().split("\n\n"):
        if block.startswith("#"):
            section = block.lstrip("# ").strip()
        elif block.strip():
            sections.append((section, " ".join(block.split())))
    AUDIO.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pipeline = KPipeline(lang_code="a", repo_id=MODEL, device="cpu")
    paragraphs = []
    tracks = [np.zeros(round(0.6 * SAMPLE_RATE), dtype=np.float32)]
    for number, (section, original) in enumerate(sections, 1):
        spoken = spoken_text(original)
        cache = AUDIO / f"paragraph_{number:02d}.json"
        path = AUDIO / f"paragraph_{number:02d}.wav"
        metadata = json.loads(cache.read_text()) if cache.exists() else {}
        if (
            path.exists()
            and metadata.get("model") == MODEL
            and metadata.get("text") == original
            and metadata.get("spoken_text") == spoken
            and metadata.get("speed") == SPEED
            and metadata.get("voice") == VOICE
        ):
            audio, rate = sf.read(path, dtype="float32")
            assert rate == SAMPLE_RATE
        else:
            results = list(pipeline(spoken, voice=VOICE, speed=SPEED, split_pattern=None))
            assert len(results) == 1, "Paragraph exceeded Kokoro context."
            result = results[0]
            audio = result.audio.numpy()
            sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
            metadata = dict(
                model=MODEL,
                text=original,
                spoken_text=spoken,
                voice=VOICE,
                speed=SPEED,
                phonemes=result.phonemes,
                words=word_timestamps(original, result.tokens, 1),
            )
            cache.write_text(json.dumps(metadata, indent=2) + "\n")
        duration = len(audio) / SAMPLE_RATE
        paragraphs.append(
            dict(metadata, number=number, section=section, duration=duration, audio_file=str(path.relative_to(ROOT)))
        )
        tracks.extend((audio, np.zeros(round(0.3 * SAMPLE_RATE), dtype=np.float32)))
        print(f"Generated paragraph {number:02d}/{len(sections)}: {duration:.2f}s, fixed speed {SPEED}", flush=True)
    readthrough = np.concatenate(tracks)
    raw_path = AUDIO / "readthrough_raw.wav"
    sf.write(raw_path, readthrough, SAMPLE_RATE, subtype="FLOAT")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(raw_path),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s24le",
            str(ROOT / "voice_over_readthrough.wav"),
        ],
        check=True,
    )
    report = dict(
        model=MODEL,
        voice=VOICE,
        speed=SPEED,
        tempo_adjustment=False,
        device="cpu",
        kokoro_version=importlib.metadata.version("kokoro"),
        python_environment=sys.executable,
        pronunciations=PRONUNCIATIONS,
        sample_rate=SAMPLE_RATE,
        word_count=sum(len(words(p["text"])) for p in paragraphs),
        speech_duration=sum(p["duration"] for p in paragraphs),
        readthrough_duration=len(readthrough) / SAMPLE_RATE,
        paragraphs=paragraphs,
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
    )
    (AUDIO / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved natural-speed voice_over_readthrough.wav. No audio time stretching.", flush=True)


def main():
    # eSpeak fails to initialize from deeply nested venv data paths. Its wrapper
    # resolves symlinks, so use a real temporary copy while generating speech.
    with TemporaryDirectory(prefix="arena_tts_") as temporary:
        data = Path(temporary) / "espeak-ng-data"
        shutil.copytree(espeakng_loader.get_data_path(), data)
        espeak.EspeakWrapper.set_data_path(str(data))
        generate_narration()


if __name__ == "__main__":
    main()
