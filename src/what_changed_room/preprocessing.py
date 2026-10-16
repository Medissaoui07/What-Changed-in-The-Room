"""Video validation and deterministic frame extraction utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2


class VideoValidationError(ValueError):
    """Raised when a video cannot be opened or lacks usable metadata or frames."""


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata read from a video container."""

    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class ExtractedFrame:
    """A frame written to disk and its source-video timestamp."""

    filename: str
    frame_index: int
    timestamp_seconds: float


def inspect_video(path: str | Path) -> VideoInfo:
    """Open a video and verify its metadata and first decodable frame."""
    video_path = Path(path)
    if not video_path.is_file():
        raise VideoValidationError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            raise VideoValidationError(f"Video has invalid frame rate ({fps}): {video_path}")
        if frame_count <= 0:
            raise VideoValidationError(f"Video reports no frames: {video_path}")
        if width <= 0 or height <= 0:
            raise VideoValidationError(f"Video has invalid dimensions ({width}x{height}): {video_path}")
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise VideoValidationError(f"Video contains no decodable frames: {video_path}")
        return VideoInfo(
            path=str(video_path.resolve()),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=frame_count / fps,
        )
    finally:
        capture.release()


def extract_video_frames(
    path: str | Path,
    output_dir: str | Path,
    *,
    sample_interval_seconds: float = 1.0,
    max_frames: int | None = None,
) -> tuple[VideoInfo, list[ExtractedFrame]]:
    """Extract approximately uniform frames and return their source timestamps.

    Sampling selects the first frame at or after each requested interval. The
    video is decoded sequentially, avoiding codec-dependent seek behavior.
    """
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be a finite number greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")

    info = inspect_video(path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(info.path)
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(f"Could not reopen video for extraction: {path}")

    extracted: list[ExtractedFrame] = []
    next_sample_time = 0.0
    frame_index = 0
    try:
        while max_frames is None or len(extracted) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / info.fps
            if timestamp + 1e-9 >= next_sample_time:
                filename = f"{Path(path).stem}_{len(extracted):06d}.jpg"
                frame_path = destination / filename
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Could not write extracted frame: {frame_path}")
                extracted.append(
                    ExtractedFrame(
                        filename=filename,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                    )
                )
                while next_sample_time <= timestamp + 1e-9:
                    next_sample_time += sample_interval_seconds
            frame_index += 1
    finally:
        capture.release()

    if not extracted:
        raise VideoValidationError(f"No frames could be extracted from video: {path}")
    return info, extracted


def preprocess_pair(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path,
    *,
    sample_interval_seconds: float = 1.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Extract frames for both videos and write a JSON frame manifest."""
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be a finite number greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")

    before_info = inspect_video(before_path)
    after_info = inspect_video(after_path)
    root = Path(output_dir)
    before_info, before_frames = extract_video_frames(
        before_path,
        root / "before",
        sample_interval_seconds=sample_interval_seconds,
        max_frames=max_frames,
    )
    after_info, after_frames = extract_video_frames(
        after_path,
        root / "after",
        sample_interval_seconds=sample_interval_seconds,
        max_frames=max_frames,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "sampling": {
            "interval_seconds": sample_interval_seconds,
            "max_frames_per_video": max_frames,
        },
        "videos": {
            "before": {
                "info": asdict(before_info),
                "frames": [asdict(frame) for frame in before_frames],
            },
            "after": {
                "info": asdict(after_info),
                "frames": [asdict(frame) for frame in after_frames],
            },
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "frame_manifest.json"
    temporary_path = root / "frame_manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest
