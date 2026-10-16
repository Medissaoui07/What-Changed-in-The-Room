from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from what_changed_room.object_geometry import _triangulate_mask_centers


def _polygon_at(x: float, y: float) -> list[list[float]]:
    return [[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2], [x - 2, y + 2]]


def test_mask_ray_triangulation_recovers_anchor_and_rejects_bad_view() -> None:
    target = np.asarray([0.25, 0.1, 8.0])
    camera = {
        "matrix": np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        "distortion": np.zeros(4),
    }
    cameras = {1: camera}
    images = {}
    annotations = []
    camera_centers = [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0]
    for index, center_x in enumerate(camera_centers):
        frame = f"before_{index:06d}.jpg"
        camera_center = np.asarray([center_x, 0.0, 0.0])
        translation = -camera_center
        images[frame] = {
            "camera_id": 1,
            "rotation": np.eye(3),
            "translation": translation,
        }
        u = 50.0 + 100.0 * (target[0] - center_x) / target[2]
        v = 50.0 + 100.0 * target[1] / target[2]
        annotations.append({
            "capture": "before", "frame": frame, "label": "chair",
            "score": 0.9, "polygon_xy": _polygon_at(u, v),
        })

    # A false detector proposal should be rejected by the ray-consensus fit.
    annotations.append({
        "capture": "before", "frame": "before_000006.jpg", "label": "chair",
        "score": 0.9, "polygon_xy": _polygon_at(95.0, 50.0),
    })
    images["before_000006.jpg"] = {
        "camera_id": 1,
        "rotation": np.eye(3),
        "translation": np.asarray([-2.5, 0.0, 0.0]),
    }

    with (
        patch("what_changed_room.object_geometry._read_colmap_cameras", return_value=cameras),
        patch("what_changed_room.object_geometry._read_colmap_images", return_value=images),
    ):
        tracks = _triangulate_mask_centers(annotations, Path("unused"), {"chair"})

    assert len(tracks) == 1
    track = tracks[0]
    assert track["diagnostics"]["view_count"] == 6
    assert track["annotation_indices"] == list(range(6))
    np.testing.assert_allclose(track["centroid_xyz_colmap"], target, atol=1e-5)
