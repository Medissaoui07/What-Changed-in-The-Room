"""Back-project reviewed image masks with accepted depth into a COLMAP map."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

from .depth import _quaternion_rotation


_CAMERA_PARAMETER_COUNTS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated COLMAP binary model")
    return data


def _read_colmap_cameras(model_dir: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    with (model_dir / "cameras.bin").open("rb") as stream:
        (count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", _read_exact(stream, 24))
            parameter_count = _CAMERA_PARAMETER_COUNTS.get(model_id)
            if parameter_count is None:
                raise ValueError(f"unsupported COLMAP camera model ID {model_id}")
            params = np.asarray(
                struct.unpack("<" + "d" * parameter_count, _read_exact(stream, parameter_count * 8)),
                dtype=np.float64,
            )
            if model_id == 0:  # SIMPLE_PINHOLE
                fx, cx, cy = params
                fy, distortion = fx, np.zeros(4, dtype=np.float64)
            elif model_id == 1:  # PINHOLE
                fx, fy, cx, cy = params
                distortion = np.zeros(4, dtype=np.float64)
            elif model_id in {2, 3, 4}:  # SIMPLE_RADIAL, RADIAL, OPENCV
                if model_id == 2:
                    fx, cx, cy, k1 = params
                    fy, distortion = fx, np.asarray([k1, 0.0, 0.0, 0.0])
                elif model_id == 3:
                    fx, cx, cy, k1, k2 = params
                    fy, distortion = fx, np.asarray([k1, k2, 0.0, 0.0])
                else:
                    fx, fy, cx, cy, k1, k2, p1, p2 = params
                    distortion = np.asarray([k1, k2, p1, p2])
            else:
                raise ValueError(
                    f"COLMAP camera model {model_id} is not supported for mask back-projection; "
                    "supported models are SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, and OPENCV"
                )
            cameras[camera_id] = {
                "width": int(width), "height": int(height),
                "matrix": np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
                "distortion": distortion,
                "model_id": model_id,
            }
    return cameras


def _read_colmap_images(model_dir: Path) -> dict[str, dict[str, object]]:
    images: dict[str, dict[str, object]] = {}
    with (model_dir / "images.bin").open("rb") as stream:
        (count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            image_id, = struct.unpack("<i", _read_exact(stream, 4))
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", _read_exact(stream, 56))
            camera_id, = struct.unpack("<i", _read_exact(stream, 4))
            name = bytearray()
            while (char := _read_exact(stream, 1)) != b"\0":
                name.extend(char)
            (point_count,) = struct.unpack("<Q", _read_exact(stream, 8))
            _read_exact(stream, int(point_count) * 24)
            images[name.decode("utf-8")] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "rotation": _quaternion_rotation(qw, qx, qy, qz),
                "translation": np.asarray([tx, ty, tz], dtype=np.float64),
            }
    return images


def _match_model_image(model_images: dict[str, dict[str, object]], capture: str, frame: str) -> str | None:
    matches = [name for name in model_images if name == frame or name.endswith("_" + frame)]
    if capture == "before":
        matches = [name for name in matches if name == frame]
    return matches[0] if len(matches) == 1 else None


def _mask_for_detection(item: dict[str, object], width: int, height: int) -> np.ndarray | None:
    polygon = item.get("polygon_xy")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    box = item.get("bbox_xyxy")
    if isinstance(box, list) and len(box) == 4:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        bounds = np.zeros_like(mask)
        bounds[max(0, y1):min(height, y2 + 1), max(0, x1):min(width, x2 + 1)] = 1
        mask &= bounds
    return mask.astype(bool)


def _triangulate_mask_centers(
    annotations: list[dict[str, object]],
    model_dir: Path,
    labels: set[str],
    *,
    minimum_views: int = 4,
    minimum_confidence: float = 0.40,
    inlier_residual_limit: float = 0.50,
    maximum_p90_residual: float = 0.48,
) -> list[dict[str, object]]:
    """Estimate conservative per-capture object anchors from registered mask rays.

    A 2D mask centroid is not a persistent physical surface point, so this is a
    provisional object-center estimate. RANSAC rejects inconsistent detections;
    the returned diagnostics let later stages gate its use.
    """
    cameras = _read_colmap_cameras(model_dir)
    model_images = _read_colmap_images(model_dir)
    rays_by_group: dict[tuple[str, str], list[dict[str, object]]] = {}
    for annotation_index, item in enumerate(annotations):
        capture, frame = item.get("capture"), item.get("frame")
        label = str(item.get("label", "unknown")).casefold()
        score = item.get("score")
        if (
            capture not in {"before", "after"}
            or not isinstance(frame, str)
            or label not in labels
            or not isinstance(score, (int, float))
            or float(score) < minimum_confidence
        ):
            continue
        model_name = _match_model_image(model_images, str(capture), frame)
        if model_name is None:
            continue
        polygon = item.get("polygon_xy")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        polygon_array = np.asarray(polygon, dtype=np.float32)
        if polygon_array.ndim != 2 or polygon_array.shape[1] != 2 or not np.isfinite(polygon_array).all():
            continue
        moments = cv2.moments(polygon_array.reshape(-1, 1, 2))
        if abs(float(moments["m00"])) < 1e-5:
            continue
        pixel = np.asarray([[[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]], dtype=np.float32)
        pose = model_images[model_name]
        camera = cameras.get(int(pose["camera_id"]))
        if camera is None:
            continue
        normalized = cv2.undistortPoints(pixel, camera["matrix"], camera["distortion"]).reshape(2)
        rotation = pose["rotation"]
        translation = pose["translation"]
        assert isinstance(rotation, np.ndarray) and isinstance(translation, np.ndarray)
        origin = -rotation.T @ translation
        direction = rotation.T @ np.asarray([normalized[0], normalized[1], 1.0])
        direction /= np.linalg.norm(direction)
        rays_by_group.setdefault((str(capture), label), []).append({
            "annotation_index": annotation_index,
            "frame": frame,
            "score": float(score),
            "origin": origin,
            "direction": direction,
        })

    result: list[dict[str, object]] = []
    identity = np.eye(3)
    for (capture, label), rays in sorted(rays_by_group.items()):
        # Avoid merging multiple same-class instances in a frame into one ray track.
        by_frame: dict[str, list[dict[str, object]]] = {}
        for ray in rays:
            by_frame.setdefault(str(ray["frame"]), []).append(ray)
        rays = [items[0] for items in by_frame.values() if len(items) == 1]
        if len(rays) < minimum_views:
            continue

        best_ids: list[int] = []
        best_residual = float("inf")
        for first in range(len(rays)):
            for second in range(first + 1, len(rays)):
                d1 = np.asarray(rays[first]["direction"])
                d2 = np.asarray(rays[second]["direction"])
                parallax = float(np.degrees(np.arccos(np.clip(abs(float(d1 @ d2)), -1.0, 1.0))))
                if parallax < 1.5:
                    continue
                pair = [rays[first], rays[second]]
                matrix = sum(identity - np.outer(np.asarray(ray["direction"]), np.asarray(ray["direction"])) for ray in pair)
                rhs = sum((identity - np.outer(np.asarray(ray["direction"]), np.asarray(ray["direction"]))) @ np.asarray(ray["origin"]) for ray in pair)
                if np.linalg.cond(matrix) > 1e7:
                    continue
                hypothesis = np.linalg.solve(matrix, rhs)
                inliers: list[int] = []
                residuals = []
                for ray_index, ray in enumerate(rays):
                    origin = np.asarray(ray["origin"])
                    direction = np.asarray(ray["direction"])
                    along = float((hypothesis - origin) @ direction)
                    residual = float(np.linalg.norm((hypothesis - origin) - along * direction))
                    if along > 0 and residual <= inlier_residual_limit:
                        inliers.append(ray_index)
                        residuals.append(residual)
                p90 = float(np.quantile(residuals, 0.9)) if residuals else float("inf")
                if len(inliers) > len(best_ids) or (len(inliers) == len(best_ids) and p90 < best_residual):
                    best_ids, best_residual = inliers, p90

        if len(best_ids) < minimum_views:
            continue
        inlier_rays = [rays[index] for index in best_ids]
        matrix = sum(identity - np.outer(np.asarray(ray["direction"]), np.asarray(ray["direction"])) for ray in inlier_rays)
        rhs = sum((identity - np.outer(np.asarray(ray["direction"]), np.asarray(ray["direction"]))) @ np.asarray(ray["origin"]) for ray in inlier_rays)
        if np.linalg.cond(matrix) > 1e7:
            continue
        center = np.linalg.solve(matrix, rhs)
        residuals = []
        for ray in inlier_rays:
            origin = np.asarray(ray["origin"])
            direction = np.asarray(ray["direction"])
            along = float((center - origin) @ direction)
            if along <= 0:
                residuals.append(float("inf"))
            else:
                residuals.append(float(np.linalg.norm((center - origin) - along * direction)))
        p90 = float(np.quantile(residuals, 0.9))
        directions = [np.asarray(ray["direction"]) for ray in inlier_rays]
        max_parallax = max(
            float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))
            for i, a in enumerate(directions) for b in directions[i + 1:]
        )
        if p90 > maximum_p90_residual or max_parallax < 3.0:
            continue
        result.append({
            "capture": capture,
            "label": label,
            "centroid_xyz_colmap": center.tolist(),
            "annotation_indices": [int(ray["annotation_index"]) for ray in inlier_rays],
            "diagnostics": {
                "method": "registered_mask_centroid_ray_intersection",
                "view_count": len(inlier_rays),
                "candidate_view_count": len(rays),
                "inlier_ratio": len(inlier_rays) / len(rays),
                "median_ray_residual_colmap_units": float(np.median(residuals)),
                "p90_ray_residual_colmap_units": p90,
                "maximum_ray_parallax_degrees": max_parallax,
                "maximum_inlier_residual_colmap_units": inlier_residual_limit,
                "status": "provisional_pass",
            },
        })
    return result


def _write_ply(path: Path, point_groups: list[dict[str, object]]) -> int:
    points = sum(len(group["xyz"]) for group in point_groups)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {points}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property int object_index\nproperty uchar capture_id\nend_header\n")
        for group in point_groups:
            xyz = group["xyz"]
            rgb = group["rgb"]
            assert isinstance(xyz, np.ndarray) and isinstance(rgb, np.ndarray)
            capture_id = 0 if group["capture"] == "before" else 1
            object_index = int(group["object_index"])
            for point, color in zip(xyz, rgb, strict=True):
                stream.write(
                    f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} "
                    f"{object_index} {capture_id}\n"
                )
    return points


def backproject_object_masks(
    annotations_path: str | Path,
    depth_report_path: str | Path,
    model_dir: str | Path,
    before_images: str | Path,
    after_images: str | Path,
    output_dir: str | Path,
    *,
    max_points_per_mask: int = 2500,
    minimum_points_per_mask: int = 20,
    triangulate_labels: tuple[str, ...] = (),
) -> dict[str, object]:
    """Back-project polygons only for frames with accepted depth calibration.

    Resulting points are in the shared COLMAP world coordinate frame and retain
    arbitrary COLMAP scale. Detector masks are explicitly marked provisional.
    """
    annotations_path, depth_report_path = Path(annotations_path), Path(depth_report_path)
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    image_dirs = {"before": Path(before_images), "after": Path(after_images)}
    annotation_data = json.loads(annotations_path.read_text(encoding="utf-8"))
    depth_data = json.loads(depth_report_path.read_text(encoding="utf-8"))
    if annotation_data.get("schema_version") != 1 or not isinstance(annotation_data.get("observations"), list):
        raise ValueError("annotations must have schema_version 1 and an observations array")
    if not isinstance(depth_data.get("frames"), list):
        raise ValueError("depth report must contain a frames array")

    depth_records: dict[tuple[str, str], dict[str, object]] = {}
    for record in depth_data["frames"]:
        if not isinstance(record, dict):
            continue
        calibration = record.get("calibration")
        if (
            isinstance(record.get("capture"), str)
            and isinstance(record.get("frame"), str)
            and isinstance(calibration, dict)
            and calibration.get("status") == "calibrated"
            and record.get("calibrated_camera_z_npy")
        ):
            depth_records[(record["capture"], record["frame"])] = record

    cameras = _read_colmap_cameras(model_dir)
    model_images = _read_colmap_images(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_points: list[dict[str, object]] = []
    output_observations: list[dict[str, object]] = []
    skipped: dict[str, int] = {}
    depth_cache: dict[tuple[str, str], np.ndarray] = {}
    image_cache: dict[tuple[str, str], np.ndarray] = {}
    selected_keys = set(depth_records)

    for annotation_index, item in enumerate(annotation_data["observations"]):
        if not isinstance(item, dict):
            continue
        capture, frame = item.get("capture"), item.get("frame")
        key = (capture, frame) if isinstance(capture, str) and isinstance(frame, str) else None
        if key not in selected_keys:
            skipped["no_accepted_depth_for_frame"] = skipped.get("no_accepted_depth_for_frame", 0) + 1
            continue
        depth_record = depth_records[key]
        image_key = key
        frame_path = image_dirs[capture] / frame
        image = image_cache.get(image_key)
        if image is None:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not read image {frame_path}")
            image_cache[image_key] = image
        height, width = image.shape[:2]
        depth = depth_cache.get(image_key)
        if depth is None:
            depth_path = Path(str(depth_record["calibrated_camera_z_npy"]))
            depth = np.load(depth_path).astype(np.float32, copy=False)
            if depth.shape != (height, width):
                raise ValueError(f"depth map {depth_path} is {depth.shape}, image is {(height, width)}")
            depth_cache[image_key] = depth

        result: dict[str, object] = {
            "object_index": len(output_observations),
            "annotation_index": annotation_index,
            "capture": capture,
            "frame": frame,
            "label": item.get("label", "unknown"),
            "bbox_xyxy": item.get("bbox_xyxy"),
            "polygon_xy": item.get("polygon_xy"),
            "score": item.get("score"),
            "source": item.get("source", "unknown"),
            "evidence_status": "provisional_mask_geometry",
            "mask_review_status": (
                "unreviewed_detector_proposal"
                if item.get("source") == "detector"
                else "manual_mask_identity_not_reviewed"
            ),
            "coordinate_frame": "before_colmap_map",
            "metric_scale": False,
            "depth_calibration_status": depth_record["calibration"].get("status"),
            "depth_calibration": depth_record["calibration"],
        }
        model_name = _match_model_image(model_images, capture, frame)
        if model_name is None:
            result.update({"geometry_status": "image_space_only", "reason": "frame_not_registered_in_shared_model"})
            output_observations.append(result)
            skipped["frame_not_registered"] = skipped.get("frame_not_registered", 0) + 1
            continue
        mask = _mask_for_detection(item, width, height)
        if mask is None:
            result.update({"geometry_status": "image_space_only", "reason": "polygon_mask_missing_or_invalid"})
            output_observations.append(result)
            skipped["missing_polygon_mask"] = skipped.get("missing_polygon_mask", 0) + 1
            continue

        valid_mask_depth = mask & np.isfinite(depth) & (depth > 0)
        valid_y, valid_x = np.where(valid_mask_depth)
        if len(valid_x) < minimum_points_per_mask:
            result.update({
                "geometry_status": "image_space_only",
                "reason": "too_few_valid_depth_pixels_in_mask",
                "valid_depth_pixels": int(len(valid_x)),
            })
            output_observations.append(result)
            skipped["too_few_depth_pixels"] = skipped.get("too_few_depth_pixels", 0) + 1
            continue
        if len(valid_x) > max_points_per_mask:
            chosen = np.linspace(0, len(valid_x) - 1, max_points_per_mask, dtype=np.int64)
            valid_x, valid_y = valid_x[chosen], valid_y[chosen]
        camera_info = cameras.get(int(model_images[model_name]["camera_id"]))
        if camera_info is None:
            raise ValueError(f"camera for registered image {model_name} is absent from cameras.bin")
        if (int(camera_info["width"]), int(camera_info["height"])) != (width, height):
            raise ValueError(f"COLMAP camera size does not match {frame}: {camera_info['width']}x{camera_info['height']} vs {width}x{height}")

        pixels = np.column_stack((valid_x.astype(np.float32), valid_y.astype(np.float32))).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(
            pixels,
            camera_info["matrix"],
            camera_info["distortion"],
        ).reshape(-1, 2)
        z = depth[valid_y, valid_x].astype(np.float64)
        camera_xyz = np.column_stack((normalized[:, 0] * z, normalized[:, 1] * z, z))
        pose = model_images[model_name]
        rotation = pose["rotation"]
        translation = pose["translation"]
        assert isinstance(rotation, np.ndarray) and isinstance(translation, np.ndarray)
        world_xyz = (rotation.T @ (camera_xyz - translation).T).T
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[valid_y, valid_x]
        object_index = len(output_observations)
        grouped_points.append({
            "object_index": object_index, "capture": capture,
            "xyz": world_xyz.astype(np.float32), "rgb": rgb,
        })
        centroid = np.median(world_xyz, axis=0)
        result.update({
            "geometry_status": "backprojected_provisional",
            "registered_image_name": model_name,
            "point_count": int(len(world_xyz)),
            "median_camera_z": float(np.median(z)),
            "centroid_xyz_colmap": centroid.tolist(),
            "bounds_min_xyz_colmap": world_xyz.min(axis=0).tolist(),
            "bounds_max_xyz_colmap": world_xyz.max(axis=0).tolist(),
            "point_cloud_object_index": object_index,
        })
        output_observations.append(result)

    triangulated_tracks = _triangulate_mask_centers(
        annotation_data["observations"], model_dir,
        {label.casefold() for label in triangulate_labels},
    ) if triangulate_labels else []
    observation_by_annotation = {
        int(item["annotation_index"]): item
        for item in output_observations
        if isinstance(item.get("annotation_index"), int)
    }
    for track in triangulated_tracks:
        for annotation_index in track["annotation_indices"]:
            annotation = annotation_data["observations"][annotation_index]
            observation = observation_by_annotation.get(annotation_index)
            if observation is None:
                observation = {
                    "object_index": len(output_observations),
                    "annotation_index": annotation_index,
                    "capture": annotation["capture"],
                    "frame": annotation["frame"],
                    "label": annotation.get("label", "unknown"),
                    "bbox_xyxy": annotation.get("bbox_xyxy"),
                    "polygon_xy": annotation.get("polygon_xy"),
                    "score": annotation.get("score"),
                    "source": annotation.get("source", "unknown"),
                    "evidence_status": "provisional_multiview_object_anchor",
                    "mask_review_status": "unreviewed_detector_proposal",
                    "coordinate_frame": "before_colmap_map",
                    "metric_scale": False,
                }
                output_observations.append(observation)
                observation_by_annotation[annotation_index] = observation
            observation.update({
                "geometry_status": "triangulated_provisional",
                "geometry_method": "registered_mask_centroid_ray_intersection",
                "centroid_xyz_colmap": track["centroid_xyz_colmap"],
                "triangulation": track["diagnostics"],
                "triangulation_track_frames": [
                    annotation_data["observations"][index]["frame"]
                    for index in track["annotation_indices"]
                ],
            })

    point_path = output_dir / "objects.ply"
    total_points = _write_ply(point_path, grouped_points)
    report = {
        "schema_version": 1,
        "shared_model": str(model_dir.resolve()),
        "depth_report": str(depth_report_path.resolve()),
        "annotation_source": str(annotations_path.resolve()),
        "coordinate_frame": "before_colmap_map",
        "metric_scale": False,
        "evidence_policy": {
            "detector_masks": "provisional; review masks and object identity before change claims",
            "depth": (
                "only frames with accepted depth calibration are back-projected"
            ),
            "triangulation": (
                "optional provisional object-center anchors from multiple registered mask-centroid rays; "
                "these are not object surfaces and do not establish metric scale"
            ),
            "missing_observations": "never prove an object was removed",
        },
        "observation_count": len(output_observations),
        "backprojected_count": sum(x.get("geometry_status") == "backprojected_provisional" for x in output_observations),
        "triangulated_count": sum(x.get("geometry_status") == "triangulated_provisional" for x in output_observations),
        "image_space_only_count": sum(x.get("geometry_status") == "image_space_only" for x in output_observations),
        "point_count": total_points,
        "point_cloud": str(point_path.resolve()),
        "observations": output_observations,
        "skipped_annotation_counts": skipped,
    }
    report_path = output_dir / "object_geometry.json"
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report
