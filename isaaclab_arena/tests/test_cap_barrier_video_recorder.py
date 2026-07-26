# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for CAP grocery exterior-camera video evidence."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.integrations.cap_barrier.video_recorder import (
    CAP_GROCERY_VIDEO_FIRST_CAPTURE_GENERATION,
    CAP_GROCERY_VIDEO_FPS,
    CAP_GROCERY_VIDEO_SAMPLE_EVERY_FRAMES,
    GroceryVideoRecorder,
    _extract_exterior_rgb,
)

_HEIGHT = 6
_WIDTH = 8


class _FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames = []
        self.closed = False

    def send(self, frame) -> None:
        if frame is not None:
            self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True
        self.path.write_bytes(b"fake-mp4")


class _FakeWriterFactory:
    def __init__(self) -> None:
        self.calls = []
        self.writers: list[_FakeWriter] = []

    def __call__(self, path, size, **kwargs):
        self.calls.append((path, size, kwargs))
        writer = _FakeWriter(path)
        self.writers.append(writer)
        return writer


def _captured_frame(frame_index: int, *, camera_name: str = "exterior_cam"):
    value = frame_index % 256
    return SimpleNamespace(
        camera_name=camera_name,
        width=_WIDTH,
        height=_HEIGHT,
        rgb=bytes([value]) * (_WIDTH * _HEIGHT * 3),
    )


def _make_recorder(
    tmp_path: Path,
    *,
    extractor=None,
    validator=None,
):
    output_path = tmp_path / "grocery.mp4"
    markers: list[str] = []
    writer_factory = _FakeWriterFactory()
    extracted: list[int] = []

    def default_extractor(_environment, *, frame_index: int):
        extracted.append(frame_index)
        return _captured_frame(frame_index)

    recorder = GroceryVideoRecorder(
        SimpleNamespace(_environment=object()),
        markers.append,
        output_path=output_path,
        frame_extractor=extractor or default_extractor,
        writer_factory=writer_factory,
        validator=validator or (lambda _path, _frames: None),
    )
    recorder.begin_generation(CAP_GROCERY_VIDEO_FIRST_CAPTURE_GENERATION)
    return recorder, output_path, markers, writer_factory, extracted


def test_recorder_skips_generation_one_and_requires_monotonic_lifecycle(tmp_path) -> None:
    output = tmp_path / "grocery.mp4"
    extracted: list[int] = []
    recorder = GroceryVideoRecorder(
        SimpleNamespace(_environment=object()),
        lambda _marker: None,
        output_path=output,
        frame_extractor=lambda _environment, *, frame_index: (
            extracted.append(frame_index) or _captured_frame(frame_index)
        ),
        writer_factory=_FakeWriterFactory(),
        validator=lambda _path, _frames: None,
    )

    with pytest.raises(RuntimeError, match="before begin_generation"):
        recorder(0)
    recorder.begin_generation(1)
    for frame in range(41):
        recorder(frame)
    assert extracted == []
    with pytest.raises(ValueError, match="must advance"):
        recorder.begin_generation(1)

    recorder.begin_generation(2)
    recorder(0)
    assert extracted == [0]
    recorder.close()


def test_rgb_extractor_reads_only_existing_exterior_camera() -> None:
    class _Tensor:
        shape = (_HEIGHT, _WIDTH, 3)

        def __getitem__(self, _index):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)

    renders: list[bool] = []
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": _Tensor()}))

    class _Scene:
        sensors = {"exterior_cam": camera}

        def __getitem__(self, name: str):
            return self.sensors[name]

    environment = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=_Scene(),
            sim=SimpleNamespace(render=lambda *, skip_app_pumping: renders.append(skip_app_pumping)),
        )
    )

    frame = _extract_exterior_rgb(environment, frame_index=7)

    assert renders == [False]
    assert frame.camera_name == "exterior_cam"
    assert (frame.width, frame.height) == (_WIDTH, _HEIGHT)
    assert len(frame.rgb) == _WIDTH * _HEIGHT * 3


def test_recorder_samples_existing_exterior_camera_at_ten_hz(tmp_path) -> None:
    recorder, output, _, writer_factory, extracted = _make_recorder(tmp_path)

    for frame in range(45):
        recorder(frame)

    assert CAP_GROCERY_VIDEO_FPS == 10
    assert CAP_GROCERY_VIDEO_SAMPLE_EVERY_FRAMES == 20
    assert extracted == [0, 1, 2]
    assert recorder.frame_count == 3
    assert len(writer_factory.writers) == 1
    assert len(writer_factory.writers[0].frames) == 3
    path, size, kwargs = writer_factory.calls[0]
    assert path == f"{output}.part"
    assert size == (_WIDTH, _HEIGHT)
    assert kwargs == {
        "pix_fmt_in": "rgb24",
        "fps": 10,
        "macro_block_size": 1,
        "output_params": ["-f", "mp4"],
    }

    recorder.close()
    assert not output.exists()
    assert not Path(f"{output}.part").exists()


def test_regular_finalize_request_validates_and_atomically_publishes(tmp_path) -> None:
    validated: list[tuple[str, int]] = []
    recorder, output, markers, writer_factory, _ = _make_recorder(
        tmp_path,
        validator=lambda path, frames: validated.append((path, frames)),
    )
    recorder(0)
    recorder(20)
    Path(f"{output}.finalize").touch()

    recorder(21)
    recorder(22)

    assert recorder.finalized is True
    assert output.read_bytes() == b"fake-mp4"
    assert not Path(f"{output}.part").exists()
    assert writer_factory.writers[0].closed is True
    assert validated == [(f"{output}.part", 2)]
    assert any(marker == f"CAP_GROCERY_VIDEO_RECORDED path={output} frames=2 fps=10" for marker in markers)
    assert Path(f"{output}.finalize").is_file()


def test_close_honors_finalize_request_without_another_physics_frame(tmp_path) -> None:
    recorder, output, _, _, _ = _make_recorder(tmp_path)
    recorder(0)
    Path(f"{output}.finalize").touch()

    recorder.close()

    assert recorder.finalized is True
    assert output.is_file()
    assert not Path(f"{output}.part").exists()


def test_close_without_finalize_request_discards_partial(tmp_path) -> None:
    recorder, output, markers, writer_factory, _ = _make_recorder(tmp_path)
    recorder(0)

    recorder.close()
    recorder.close()

    assert writer_factory.writers[0].closed is True
    assert not output.exists()
    assert not Path(f"{output}.part").exists()
    assert markers[-1].endswith("frames=1 reason=no-finalize-request")


@pytest.mark.parametrize("stale_suffix", ("", ".part", ".finalize"))
def test_recorder_refuses_stale_or_competing_paths(tmp_path, stale_suffix: str) -> None:
    output = tmp_path / "grocery.mp4"
    Path(f"{output}{stale_suffix}").touch()

    with pytest.raises(FileExistsError, match="already exists"):
        GroceryVideoRecorder(
            SimpleNamespace(_environment=object()),
            lambda _marker: None,
            output_path=output,
        )


@pytest.mark.parametrize(
    ("output_path", "message"),
    [
        ("relative.mp4", "must be absolute"),
        ("/tmp/video.avi", "must end in .mp4"),
    ],
)
def test_recorder_rejects_invalid_output_path(output_path: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GroceryVideoRecorder(
            SimpleNamespace(_environment=object()),
            lambda _marker: None,
            output_path=output_path,
        )


def test_recorder_requires_existing_output_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GroceryVideoRecorder(
            SimpleNamespace(_environment=object()),
            lambda _marker: None,
            output_path=tmp_path / "missing" / "video.mp4",
        )


def test_finalize_request_must_be_a_regular_file_and_partial_is_discarded(tmp_path) -> None:
    recorder, output, _, writer_factory, _ = _make_recorder(tmp_path)
    recorder(0)
    Path(f"{output}.finalize").mkdir()

    with pytest.raises(RuntimeError, match="not a regular file"):
        recorder(1)

    recorder.close()
    recorder.finalize_request_path.rmdir()
    assert writer_factory.writers[0].closed is True
    assert not output.exists()
    assert not Path(f"{output}.part").exists()


@pytest.mark.parametrize(
    "captured",
    [
        _captured_frame(0, camera_name="wrist_camera"),
        SimpleNamespace(
            camera_name="exterior_cam",
            width=_WIDTH,
            height=_HEIGHT,
            rgb=b"short",
        ),
    ],
)
def test_wrong_camera_or_payload_fails_before_opening_encoder(tmp_path, captured) -> None:
    recorder, output, _, writer_factory, _ = _make_recorder(
        tmp_path,
        extractor=lambda _environment, *, frame_index: captured,
    )

    with pytest.raises((RuntimeError, ValueError)):
        recorder(0)

    assert writer_factory.writers == []
    recorder.close()
    assert not output.exists()


def test_failed_validation_never_publishes_mp4(tmp_path) -> None:
    def fail_validation(_path: str, _frames: int) -> None:
        raise RuntimeError("truncated stream")

    recorder, output, _, _, _ = _make_recorder(
        tmp_path,
        validator=fail_validation,
    )
    recorder(0)
    Path(f"{output}.finalize").touch()

    with pytest.raises(RuntimeError, match="truncated stream"):
        recorder(1)

    assert not output.exists()
    assert not Path(f"{output}.part").exists()


def test_post_link_durability_failure_rolls_back_published_mp4(tmp_path, monkeypatch) -> None:
    import isaaclab_arena.integrations.cap_barrier.video_recorder as recorder_module

    fsync_calls = 0
    real_fsync = recorder_module.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("synthetic directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(recorder_module.os, "fsync", fail_directory_fsync)
    recorder, output, _, _, _ = _make_recorder(tmp_path)
    recorder(0)
    Path(f"{output}.finalize").touch()

    with pytest.raises(OSError, match="directory fsync failure"):
        recorder(1)

    assert fsync_calls == 2
    assert not output.exists()
    assert not Path(f"{output}.part").exists()
    assert recorder.finalized is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_real_encoder_produces_valid_mp4_only_after_request(tmp_path) -> None:
    output = tmp_path / "real-grocery.mp4"
    recorder = GroceryVideoRecorder(
        SimpleNamespace(_environment=object()),
        lambda _marker: None,
        output_path=output,
        frame_extractor=lambda _environment, *, frame_index: _captured_frame(frame_index),
    )
    recorder.begin_generation(CAP_GROCERY_VIDEO_FIRST_CAPTURE_GENERATION)
    for frame in range(81):
        recorder(frame)

    assert not output.exists()

    Path(f"{output}.finalize").touch()
    recorder(82)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert not Path(f"{output}.part").exists()
