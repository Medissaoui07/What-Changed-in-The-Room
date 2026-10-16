import sqlite3
import struct
from pathlib import Path

import numpy as np

from what_changed_room.colmap_sparse import (
    load_colmap_sparse_model,
    match_colmap_sparse_models,
)
from what_changed_room.registration import SimilarityTransform, register_point_clouds


def _write_model(directory: Path, point_ids: np.ndarray, points: np.ndarray,
                 descriptors: np.ndarray, image_name: str) -> None:
    directory.mkdir(parents=True)
    with (directory / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(point_ids)))
        for index, (point_id, point) in enumerate(zip(point_ids, points, strict=True)):
            stream.write(struct.pack(
                "<QdddBBBdQ", int(point_id), *point, 120, 140, 160, 0.25, 1,
            ))
            stream.write(struct.pack("<ii", 1, index))

    with (directory / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<i4d3di", 1, 1.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 1))
        stream.write(image_name.encode() + b"\0")
        stream.write(struct.pack("<Q", len(point_ids)))
        for index, point_id in enumerate(point_ids):
            stream.write(struct.pack("<ddq", float(index), float(index + 1), int(point_id)))


def _write_database(path: Path, descriptors: np.ndarray) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE descriptors (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
        )
        connection.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (1, len(descriptors), descriptors.shape[1], descriptors.tobytes()),
        )


def test_colmap_features_link_independent_models_for_registration(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    source_points = rng.uniform(-1.0, 1.0, size=(16, 3))
    expected = SimilarityTransform(
        1.35,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([0.4, -0.2, 0.7]),
    )
    target_points = expected.apply(source_points)
    descriptors = rng.integers(0, 256, size=(16, 128), dtype=np.uint8)
    before_ids = np.arange(100, 116, dtype=np.uint64)
    after_ids = np.arange(900, 916, dtype=np.uint64)

    before_model, after_model = tmp_path / "before_model", tmp_path / "after_model"
    _write_model(before_model, before_ids, target_points, descriptors, "before.jpg")
    _write_model(after_model, after_ids, source_points, descriptors, "after.jpg")
    before_db, after_db = tmp_path / "before.db", tmp_path / "after.db"
    _write_database(before_db, descriptors)
    _write_database(after_db, descriptors)

    before = load_colmap_sparse_model(before_model, before_db)
    after = load_colmap_sparse_model(after_model, after_db)
    pairs = match_colmap_sparse_models(after, before)
    result = register_point_clouds(
        after.point_cloud, before.point_cloud, pairs,
        inlier_threshold=1e-5, icp_iterations=0,
    )

    assert len(pairs) == len(source_points)
    assert result.diagnostics.success
    assert result.transform is not None
    assert np.isclose(result.transform.scale, expected.scale, atol=1e-8)
    assert np.allclose(result.transform.rotation, expected.rotation, atol=1e-8)
    assert np.allclose(result.transform.translation, expected.translation, atol=1e-8)
