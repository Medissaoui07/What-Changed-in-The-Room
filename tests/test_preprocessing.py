"""Tests for validation, extraction, manifest output, and the CLI."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from what_changed_room.cli import main
from what_changed_room.preprocessing import (
    VideoValidationError,
    extract_video_frames,
    inspect_video,
    preprocess_pair,
)


def write_test_video(path: Path, frame_count: int = 10, fps: float = 10.0) -> Path:
    """Create a small deterministic MJPEG video for preprocessing tests."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24)
    )
    if not writer.isOpened():
        pytest.fail("OpenCV could not initialize the MJPEG test video writer")
    for index in range(frame_count):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, index % 3] = index * 20
        writer.write(frame)
    writer.release()
    return path


def test_inspect_video_reads_metadata(tmp_path: Path) -> None:
    video = write_test_video(tmp_path / "sample.avi")

    info = inspect_video(video)

    assert info.fps == pytest.approx(10.0)
    assert info.frame_count == 10
    assert (info.width, info.height) == (32, 24)
    assert info.duration_seconds == pytest.approx(1.0)


def test_extract_uses_interval_and_max_frames(tmp_path: Path) -> None:
    video = write_test_video(tmp_path / "sample.avi")

    _, frames = extract_video_frames(
        video,
        tmp_path / "frames",
        sample_interval_seconds=0.3,
        max_frames=3,
    )

    assert [frame.frame_index for frame in frames] == [0, 3, 6]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.0, 0.3, 0.6])
    assert all((tmp_path / "frames" / frame.filename).is_file() for frame in frames)


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_extract_rejects_invalid_interval(tmp_path: Path, interval: float) -> None:
    with pytest.raises(ValueError, match="sample_interval_seconds"):
        extract_video_frames("missing.avi", tmp_path, sample_interval_seconds=interval)


def test_inspect_rejects_missing_or_unreadable_video(tmp_path: Path) -> None:
    with pytest.raises(VideoValidationError, match="does not exist"):
        inspect_video(tmp_path / "missing.avi")

    unreadable = tmp_path / "not_a_video.avi"
    unreadable.write_text("not video data", encoding="utf-8")
    with pytest.raises(VideoValidationError, match="Could not open video"):
        inspect_video(unreadable)


def test_preprocess_pair_writes_manifest(tmp_path: Path) -> None:
    before = write_test_video(tmp_path / "before.avi")
    after = write_test_video(tmp_path / "after.avi", frame_count=7)

    manifest = preprocess_pair(
        before,
        after,
        tmp_path / "output",
        sample_interval_seconds=0.5,
    )
    saved = json.loads((tmp_path / "output" / "frame_manifest.json").read_text())

    assert saved == manifest
    assert [row["frame_index"] for row in manifest["videos"]["before"]["frames"]] == [0, 5]
    assert [row["frame_index"] for row in manifest["videos"]["after"]["frames"]] == [0, 5]
    assert (tmp_path / "output" / "before" / "before_000000.jpg").is_file()
    assert (tmp_path / "output" / "after" / "after_000001.jpg").is_file()


def test_cli_success_and_invalid_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = write_test_video(tmp_path / "before.avi")
    after = write_test_video(tmp_path / "after.avi")
    output = tmp_path / "result"

    result = main(
        ["preprocess", str(before), str(after), "--output-dir", str(output), "--interval", "0.5"]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["before_frames"] == 2

    result = main(
        [
            "preprocess",
            str(tmp_path / "missing.avi"),
            str(after),
            "--output-dir",
            str(tmp_path / "bad"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "does not exist" in captured.err
