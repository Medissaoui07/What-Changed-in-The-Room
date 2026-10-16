"""Read COLMAP 3.x sparse models and derive cross-model 3D feature matches.

The model point IDs are local to each reconstruction. This module maps SIFT
descriptors from each model's registered image observations onto their local
3D points, then matches descriptors between models to create usable 3D pairs.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .reconstruction import PointCloud


@dataclass(frozen=True)
class ColmapSparseModel:
    """Sparse points and aggregated SIFT features linked to local 3D points."""

    point_cloud: PointCloud
    point3d_ids: NDArray[np.uint64]
    descriptors: NDArray[np.uint8]
    descriptor_point_indices: NDArray[np.int64]
    registered_image_names: tuple[str, ...]


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated COLMAP binary model")
    return value


def _read_images(path: Path) -> tuple[dict[int, NDArray[np.int64]], tuple[str, ...]]:
    """Return registered image IDs to their 2D-keypoint point3D IDs."""
    image_points: dict[int, NDArray[np.int64]] = {}
    names: list[str] = []
    with path.open("rb") as stream:
        image_count, = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(image_count):
            image_id, = struct.unpack("<i", _read_exact(stream, 4))
            _read_exact(stream, 7 * 8)  # quaternion and translation
            _read_exact(stream, 4)  # camera ID
            name_bytes = bytearray()
            while (byte := _read_exact(stream, 1)) != b"\0":
                name_bytes.extend(byte)
            name = name_bytes.decode("utf-8")
            point_count, = struct.unpack("<Q", _read_exact(stream, 8))
            point_ids = np.empty(point_count, dtype=np.int64)
            for index in range(point_count):
                _read_exact(stream, 2 * 8)  # image x,y
                point_ids[index], = struct.unpack("<q", _read_exact(stream, 8))
            image_points[image_id] = point_ids
            names.append(name)
    return image_points, tuple(names)


def _read_points(path: Path) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            point_id, x, y, z, _r, _g, _b, _error, track_length = struct.unpack(
                "<QdddBBBdQ", _read_exact(stream, struct.calcsize("<QdddBBBdQ"))
            )
            _read_exact(stream, int(track_length) * struct.calcsize("<ii"))
            ids.append(point_id)
            xyz.append((x, y, z))
    if not xyz:
        raise ValueError(f"COLMAP sparse model has no 3D points: {path.parent}")
    return np.asarray(ids, dtype=np.uint64), np.asarray(xyz, dtype=np.float64)


def load_colmap_sparse_model(model_dir: str | Path, database_path: str | Path) -> ColmapSparseModel:
    """Load a binary COLMAP 3.x model and aggregate tracked SIFT descriptors.

    ``model_dir`` must contain ``images.bin`` and ``points3D.bin``; the matching
    COLMAP database must contain the original 128-dimensional SIFT descriptors.
    Text-only exports and newer frame-based formats are rejected explicitly.
    """
    model_dir = Path(model_dir)
    database_path = Path(database_path)
    images_path, points_path = model_dir / "images.bin", model_dir / "points3D.bin"
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError(
            f"expected COLMAP 3.x binary model files images.bin and points3D.bin in {model_dir}"
        )
    if not database_path.is_file():
        raise FileNotFoundError(f"COLMAP database not found: {database_path}")

    point_ids, xyz = _read_points(points_path)
    point_index = {int(point_id): index for index, point_id in enumerate(point_ids)}
    image_points, image_names = _read_images(images_path)
    descriptor_sums = np.zeros((len(point_ids), 128), dtype=np.float64)
    descriptor_counts = np.zeros(len(point_ids), dtype=np.int32)

    try:
        with sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True) as connection:
            for image_id, point3d_for_keypoint in image_points.items():
                row = connection.execute(
                    "SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,)
                ).fetchone()
                if row is None:
                    continue
                rows, cols, blob = row
                if cols != 128 or rows != len(point3d_for_keypoint):
                    raise ValueError(
                        f"descriptor/keypoint mismatch for COLMAP image ID {image_id}: "
                        f"{rows}x{cols} descriptors, {len(point3d_for_keypoint)} observations"
                    )
                descriptors = np.frombuffer(blob, dtype=np.uint8).reshape(rows, cols)
                valid = point3d_for_keypoint >= 0
                if not np.any(valid):
                    continue
                local_indices = np.fromiter(
                    (point_index.get(int(value), -1) for value in point3d_for_keypoint[valid]),
                    dtype=np.int64,
                    count=int(valid.sum()),
                )
                mapped = local_indices >= 0
                if not np.any(mapped):
                    continue
                selected_indices = local_indices[mapped]
                selected_descriptors = descriptors[valid][mapped]
                np.add.at(descriptor_sums, selected_indices, selected_descriptors)
                np.add.at(descriptor_counts, selected_indices, 1)
    except sqlite3.Error as exc:
        raise ValueError(f"could not read COLMAP descriptor database {database_path}: {exc}") from exc
    observed = descriptor_counts > 0
    all_descriptors = np.rint(descriptor_sums[observed] / descriptor_counts[observed, None])
    all_point_indices = np.flatnonzero(observed).astype(np.int64)
    return ColmapSparseModel(
        PointCloud(xyz),
        point_ids,
        all_descriptors.astype(np.uint8),
        all_point_indices,
        image_names,
    )


def _ratio_match(query: NDArray[np.uint8], train: NDArray[np.uint8], ratio: float) -> dict[int, int]:
    if len(query) == 0 or len(train) < 2:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    results = matcher.knnMatch(query, train, k=2)
    matches: dict[int, int] = {}
    for neighbors in results:
        if len(neighbors) == 2 and neighbors[0].distance < ratio * neighbors[1].distance:
            matches[neighbors[0].queryIdx] = neighbors[0].trainIdx
    return matches


def match_colmap_sparse_models(
    source: ColmapSparseModel,
    target: ColmapSparseModel,
    *,
    ratio: float = 0.8,
) -> NDArray[np.int64]:
    """Return ``(source_point_index, target_point_index)`` mutual SIFT matches."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be in (0, 1)")
    if len(source.descriptors) < 2 or len(target.descriptors) < 2:
        return np.empty((0, 2), dtype=np.int64)
    forward = _ratio_match(source.descriptors, target.descriptors, ratio)
    reverse = _ratio_match(target.descriptors, source.descriptors, ratio)
    votes: dict[tuple[int, int], int] = {}
    for source_descriptor, target_descriptor in forward.items():
        if reverse.get(target_descriptor) != source_descriptor:
            continue
        source_point = int(source.descriptor_point_indices[source_descriptor])
        target_point = int(target.descriptor_point_indices[target_descriptor])
        key = (source_point, target_point)
        votes[key] = votes.get(key, 0) + 1
    if not votes:
        return np.empty((0, 2), dtype=np.int64)
    ranked = sorted(
        ((-count, pair[0], pair[1]) for pair, count in votes.items())
    )
    pairs: list[tuple[int, int]] = []
    used_source: set[int] = set()
    used_target: set[int] = set()
    for _vote_count, source_point, target_point in ranked:
        if source_point not in used_source and target_point not in used_target:
            pairs.append((source_point, target_point))
            used_source.add(source_point)
            used_target.add(target_point)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
