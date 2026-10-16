"""Object observations on RGB frames with optional links to sparse COLMAP points.

This module deliberately records image-space evidence even when a frame is not
registered or its region contains no mapped points. It does not infer missing
geometry, object absence, or cross-capture changes.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detector or annotator result in source-image pixel coordinates."""

    label: str
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: tuple[tuple[float, float], ...] | None = None
    score: float | None = None
    track_id: str | None = None
    source: str = "manual"


class ObjectDetector(Protocol):
    """Optional detector contract; implementations may use any local model."""

    def detect(self, image_path: Path) -> Sequence[Detection]:
        """Return detections for one RGB image, without assigning 3D depth."""


class UltralyticsSegDetector:
    """YOLO instance segmentation adapter, imported lazily so it stays optional."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n-seg.pt",
        *,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str = "auto",
        preview_dir: str | Path | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if device == "auto":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required for YOLO inference") from exc
            device = "0" if torch.cuda.is_available() else "cpu"
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO inference requires the optional 'ultralytics' package"
            ) from exc
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.preview_dir = None if preview_dir is None else Path(preview_dir)

    def detect(self, image_path: Path) -> Sequence[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        if self.preview_dir is not None:
            self.preview_dir.mkdir(parents=True, exist_ok=True)
            preview = result.plot()
            if not cv2.imwrite(str(self.preview_dir / image_path.name), preview):
                raise OSError(f"could not write YOLO preview for {image_path}")

        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            label = str(self.model.names[class_id])
            bbox = tuple(float(value) for value in box.xyxy[0].tolist())
            polygon = None
            if result.masks is not None and index < len(result.masks.xy):
                vertices = np.asarray(result.masks.xy[index], dtype=np.float32)
                if len(vertices) >= 3:
                    polygon = tuple((float(x), float(y)) for x, y in vertices)
            detections.append(Detection(
                label=label,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                polygon_xy=polygon,
                score=float(box.conf.item()),
                source="detector",
            ))
        return detections


def annotations_from_detector(
    detector: ObjectDetector,
    selected_frames: dict[str, Sequence[str | Path]],
) -> dict[str, object]:
    """Adapt an optional local detector to the same JSON input as manual labels.

    ``selected_frames`` maps only ``before`` and/or ``after`` to the RGB frames
    chosen for processing. The detector remains caller-supplied and optional.
    """
    observations: list[dict[str, object]] = []
    for capture, frames in selected_frames.items():
        if capture not in {"before", "after"}:
            raise ValueError("detector frame groups must be named 'before' or 'after'")
        for frame_value in frames:
            frame = Path(frame_value)
            if not frame.is_file():
                raise FileNotFoundError(f"detector frame does not exist: {frame}")
            for detection in detector.detect(frame):
                observations.append({
                    "capture": capture,
                    "frame": frame.name,
                    "label": detection.label,
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "polygon_xy": None if detection.polygon_xy is None else [list(point) for point in detection.polygon_xy],
                    "score": detection.score,
                    "track_id": detection.track_id,
                    "source": "detector",
                })
    return {"schema_version": 1, "observations": observations}


@dataclass(frozen=True)
class _ImageObservation:
    capture: str
    frame: str
    timestamp_seconds: float | None
    detection: Detection
    width: int
    height: int


def _read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated COLMAP binary model")
    return data


def _read_sparse_image_observations(model_dir: Path) -> tuple[dict[str, list[tuple[float, float, int]]], dict[int, tuple[float, float, float]]]:
    """Read image 2D keypoints and their existing COLMAP point IDs."""
    images_path = model_dir / "images.bin"
    points_path = model_dir / "points3D.bin"
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError(f"expected images.bin and points3D.bin in {model_dir}")

    image_observations: dict[str, list[tuple[float, float, int]]] = {}
    with images_path.open("rb") as stream:
        (image_count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(image_count):
            _image_id, = struct.unpack("<i", _read_exact(stream, 4))
            _read_exact(stream, 7 * 8)  # quaternion and translation
            _read_exact(stream, 4)  # camera ID
            name = bytearray()
            while (char := _read_exact(stream, 1)) != b"\0":
                name.extend(char)
            (keypoint_count,) = struct.unpack("<Q", _read_exact(stream, 8))
            rows: list[tuple[float, float, int]] = []
            for _ in range(keypoint_count):
                x, y, point_id = struct.unpack("<ddq", _read_exact(stream, 24))
                rows.append((x, y, point_id))
            image_observations[name.decode("utf-8")] = rows

    point_xyz: dict[int, tuple[float, float, float]] = {}
    with points_path.open("rb") as stream:
        (point_count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(point_count):
            point_id, x, y, z, _r, _g, _b, _error, track_length = struct.unpack(
                "<QdddBBBdQ", _read_exact(stream, struct.calcsize("<QdddBBBdQ"))
            )
            _read_exact(stream, int(track_length) * 8)
            point_xyz[int(point_id)] = (x, y, z)
    return image_observations, point_xyz


def _load_manifest_timestamps(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return {
            (capture, frame["filename"]): float(frame["timestamp_seconds"])
            for capture in ("before", "after")
            for frame in manifest["videos"][capture]["frames"]
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frame manifest {path}: {exc}") from exc


def _load_annotations(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read annotations JSON {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("annotations JSON must be an object with schema_version: 1")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("annotations JSON must contain an observations array")
    if not all(isinstance(item, dict) for item in observations):
        raise ValueError("each observations entry must be an object")
    return observations


def _make_detection(item: dict[str, object]) -> Detection:
    label = item.get("label")
    box = item.get("bbox_xyxy")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("each observation needs a non-empty label")
    if not isinstance(box, list | tuple) or len(box) != 4:
        raise ValueError(f"observation {label!r} needs bbox_xyxy: [x1, y1, x2, y2]")
    try:
        bbox = tuple(float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"observation {label!r} has a non-numeric bounding box") from exc
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"observation {label!r} bounding box must be finite")
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"observation {label!r} has an invalid xyxy bounding box")

    polygon_value = item.get("polygon_xy")
    polygon = None
    if polygon_value is not None:
        if not isinstance(polygon_value, list | tuple) or len(polygon_value) < 3:
            raise ValueError(f"observation {label!r} polygon_xy must contain at least 3 points")
        try:
            polygon = tuple((float(point[0]), float(point[1])) for point in polygon_value)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"observation {label!r} has an invalid polygon_xy") from exc
        if not all(math.isfinite(coord) for point in polygon for coord in point):
            raise ValueError(f"observation {label!r} polygon coordinates must be finite")

    score_value = item.get("score")
    score = None if score_value is None else float(score_value)
    if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError(f"observation {label!r} score must be between 0 and 1")
    source = item.get("source", "manual")
    if not isinstance(source, str) or source not in {"manual", "detector"}:
        raise ValueError(f"observation {label!r} source must be 'manual' or 'detector'")
    track_id = item.get("track_id")
    if track_id is not None and (not isinstance(track_id, str) or not track_id.strip()):
        raise ValueError(f"observation {label!r} track_id must be a non-empty string")
    return Detection(
        label=label.strip(),
        bbox_xyxy=bbox,  # type: ignore[arg-type]
        polygon_xy=polygon,
        score=score,
        track_id=track_id,
        source=source,
    )


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-12)


def _assign_tracks(observations: list[_ImageObservation]) -> list[dict[str, object]]:
    """Use supplied per-capture IDs or conservative adjacent-frame box overlap."""
    ordered = sorted(
        enumerate(observations),
        key=lambda pair: (
            pair[1].capture,
            float("inf") if pair[1].timestamp_seconds is None else pair[1].timestamp_seconds,
            pair[1].frame,
            pair[0],
        ),
    )
    tracks: dict[tuple[str, str], dict[str, object]] = {}
    results: list[dict[str, object] | None] = [None] * len(observations)
    counters = {"before": 0, "after": 0}
    for original_index, observation in ordered:
        detection = observation.detection
        supplied_id = detection.track_id
        method = "manual_track_id" if supplied_id is not None else "single_observation"
        assigned_id = supplied_id
        if supplied_id is None and observation.timestamp_seconds is not None:
            choices: list[tuple[float, str]] = []
            for (capture, track_id), track in tracks.items():
                last = track["last"]
                assert isinstance(last, _ImageObservation)
                if capture != observation.capture or last.frame == observation.frame:
                    continue
                if last.detection.label.casefold() != detection.label.casefold():
                    continue
                if last.timestamp_seconds is None:
                    continue
                gap = observation.timestamp_seconds - last.timestamp_seconds
                if gap < 0 or gap > 2.5:
                    continue
                overlap = _iou(last.detection.bbox_xyxy, detection.bbox_xyxy)
                if overlap >= 0.5:
                    choices.append((overlap, track_id))
            if choices:
                _overlap, assigned_id = max(choices)
                method = "provisional_bbox_iou"
        if assigned_id is None:
            counters[observation.capture] += 1
            assigned_id = f"auto_{counters[observation.capture]:04d}"
        # Capture namespace prevents accidental before/after identity claims.
        track_key = (observation.capture, assigned_id)
        track = tracks.setdefault(track_key, {"track_id": f"{observation.capture}:{assigned_id}", "observations": []})
        entries = track["observations"]
        assert isinstance(entries, list)
        entry = {
            "observation_id": f"{observation.capture}:{observation.frame}:{original_index:04d}",
            "capture": observation.capture,
            "frame": observation.frame,
            "timestamp_seconds": observation.timestamp_seconds,
            "label": detection.label,
            "bbox_xyxy": list(detection.bbox_xyxy),
            "polygon_xy": None if detection.polygon_xy is None else [list(point) for point in detection.polygon_xy],
            "image_size_wh": [observation.width, observation.height],
            "annotation_source": detection.source,
            "score": detection.score,
            "track_id": track["track_id"],
            "track_method": method,
        }
        entries.append(entry)
        track["last"] = observation
        results[original_index] = entry
    # Omit mutable tracker internals from serialized representation.
    return [entry for entry in results if entry is not None]


def _point_in_region(
    x: float,
    y: float,
    detection: Detection,
) -> bool:
    x1, y1, x2, y2 = detection.bbox_xyxy
    if x < x1 or x > x2 or y < y1 or y > y2:
        return False
    if detection.polygon_xy is None:
        return True
    contour = np.asarray(detection.polygon_xy, dtype=np.float32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def build_observation_report(
    annotations_path: str | Path,
    shared_model: str | Path,
    before_images: str | Path,
    after_images: str | Path,
    *,
    frame_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Validate annotations, track within captures, and attach observed 3D points.

    Frames without a registered COLMAP camera and boxes without observed 3D
    keypoints are retained as image-space evidence. No depth is estimated.
    """
    annotations_path = Path(annotations_path).resolve()
    model_dir = Path(shared_model).resolve()
    capture_dirs = {
        "before": Path(before_images).resolve(),
        "after": Path(after_images).resolve(),
    }
    if not annotations_path.is_file():
        raise FileNotFoundError(f"annotations file does not exist: {annotations_path}")
    for capture, directory in capture_dirs.items():
        if not directory.is_dir():
            raise NotADirectoryError(f"{capture} image directory does not exist: {directory}")
    image_map, point_xyz = _read_sparse_image_observations(model_dir)
    timestamps = _load_manifest_timestamps(None if frame_manifest is None else Path(frame_manifest))

    frame_lookup: dict[str, dict[str, Path]] = {}
    for capture, directory in capture_dirs.items():
        frame_lookup[capture] = {}
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if path.name in frame_lookup[capture]:
                    raise ValueError(f"duplicate frame filename in {capture} images: {path.name}")
                frame_lookup[capture][path.name] = path

    parsed: list[_ImageObservation] = []
    raw_annotations = _load_annotations(annotations_path)
    for index, item in enumerate(raw_annotations):
        capture = item.get("capture")
        frame = item.get("frame")
        if not isinstance(capture, str) or capture not in capture_dirs:
            raise ValueError(f"observation {index} capture must be 'before' or 'after'")
        if not isinstance(frame, str) or Path(frame).name != frame:
            raise ValueError(f"observation {index} frame must be a filename in the selected frame directory")
        image_path = frame_lookup[capture].get(frame)
        if image_path is None:
            raise FileNotFoundError(f"observation {index} frame not found in {capture} images: {frame}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read annotated frame: {image_path}")
        height, width = image.shape[:2]
        detection = _make_detection(item)
        x1, y1, x2, y2 = detection.bbox_xyxy
        if x2 > width or y2 > height:
            raise ValueError(f"observation {index} box exceeds image bounds ({width}x{height})")
        if detection.polygon_xy and any(x < 0 or y < 0 or x > width or y > height for x, y in detection.polygon_xy):
            raise ValueError(f"observation {index} polygon exceeds image bounds ({width}x{height})")
        raw_timestamp = item.get("timestamp_seconds")
        timestamp = (
            float(raw_timestamp)
            if raw_timestamp is not None
            else timestamps.get((capture, frame))
        )
        if timestamp is not None and (not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError(f"observation {index} timestamp_seconds must be finite and non-negative")
        parsed.append(_ImageObservation(capture, frame, timestamp, detection, width, height))

    tracked = _assign_tracks(parsed)
    # Resolve model camera names to frame filenames. The after localizer prefixes
    # its source image names, so exact names and unique suffix matches are valid.
    model_names_by_frame: dict[tuple[str, str], str] = {}
    for capture in capture_dirs:
        for frame in frame_lookup[capture]:
            matches = [name for name in image_map if name == frame or name.endswith("_" + frame)]
            # An after frame can occur in the map only if COLMAP registered it.
            if capture == "before":
                matches = [name for name in matches if name == frame]
            if len(matches) == 1:
                model_names_by_frame[(capture, frame)] = matches[0]

    results: list[dict[str, object]] = []
    for parsed_observation, observation in zip(parsed, tracked, strict=True):
        model_name = model_names_by_frame.get((parsed_observation.capture, parsed_observation.frame))
        anchors: dict[int, tuple[float, float, float]] = {}
        if model_name is not None:
            for x, y, point_id in image_map[model_name]:
                if point_id >= 0 and _point_in_region(x, y, parsed_observation.detection):
                    xyz = point_xyz.get(point_id)
                    if xyz is not None:
                        anchors[point_id] = xyz
        observation["registered_image_name"] = model_name
        observation["geometry_evidence"] = "sparse_keypoints_in_region" if anchors else "image_space_only"
        observation["coordinate_frame"] = "before_colmap_map" if anchors else None
        observation["linked_3d_points"] = [
            {"point3d_id": point_id, "xyz": list(xyz)}
            for point_id, xyz in sorted(anchors.items())
        ]
        observation["linked_3d_point_count"] = len(anchors)
        results.append(observation)

    grouped: dict[str, list[dict[str, object]]] = {}
    for observation in results:
        grouped.setdefault(str(observation["track_id"]), []).append(observation)
    tracks = [
        {
            "track_id": track_id,
            "capture": observations[0]["capture"],
            "label": observations[0]["label"],
            "tracking_status": (
                "manual" if all(item["track_method"] == "manual_track_id" for item in observations)
                else "provisional" if any(item["track_method"] == "provisional_bbox_iou" for item in observations)
                else "single_observation"
            ),
            "observations": observations,
        }
        for track_id, observations in sorted(grouped.items())
    ]
    return {
        "schema_version": 1,
        "coordinate_frame": "before_colmap_map",
        "shared_model": str(model_dir),
        "evidence_policy": {
            "unsupported_geometry": "keep as image_space_only; do not estimate depth",
            "linked_points": "observed sparse keypoints in the annotated region; not object surfaces",
            "missing_sparse_points": "never evidence of object removal",
            "cross_capture_change_classification": "not performed in Phase 4",
        },
        "observation_count": len(results),
        "track_count": len(tracks),
        "tracks": tracks,
    }


def write_observation_report(
    report: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Write a Phase 4 evidence report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
