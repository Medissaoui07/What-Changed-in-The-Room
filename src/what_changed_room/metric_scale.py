"""Estimate a per-reconstruction metric scale from a known planar reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def calibrate_scale_from_known_object(
    geometry_report: str | Path,
    *,
    label: str = "tv",
    width_m: float = 0.90,
    height_m: float = 0.55,
    minimum_points: int = 100,
) -> dict[str, object]:
    """Fit meters per COLMAP unit from repeated back-projected planar masks.

    Each selected mask's 3D points are robustly projected onto their first two
    PCA axes. The 5th-to-95th percentile extents are matched to a known width
    and height. Two or more consistent views are required.
    """
    if width_m <= 0 or height_m <= 0:
        raise ValueError("reference dimensions must be positive")
    report_path = Path(geometry_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    point_path = Path(str(report["point_cloud"]))
    with point_path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("metric scale needs the generated ASCII object PLY")
        vertex_count = None
        property_names: list[str] = []
        while True:
            row = stream.readline()
            if not row:
                raise ValueError("invalid object PLY header")
            fields = row.split()
            if len(fields) == 3 and fields[0] == "element" and fields[1] == "vertex":
                vertex_count = int(fields[2])
            elif len(fields) == 3 and fields[0] == "property" and vertex_count is not None:
                property_names.append(fields[2])
            elif fields and fields[0] == "end_header":
                break
        points = np.loadtxt(stream, max_rows=vertex_count, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    columns = {name: index for index, name in enumerate(property_names)}
    needed = {"x", "y", "z", "object_index"}
    if not needed.issubset(columns):
        raise ValueError("object PLY does not contain XYZ and object_index fields")

    estimates = []
    for item in report.get("observations", []):
        if (
            item.get("capture") != "before"
            or str(item.get("label", "")).casefold() != label.casefold()
            or item.get("geometry_status") != "backprojected_provisional"
        ):
            continue
        index = int(item["object_index"])
        cloud = points[points[:, columns["object_index"]].astype(np.int64) == index]
        if len(cloud) < minimum_points:
            continue
        xyz = cloud[:, [columns["x"], columns["y"], columns["z"]]]
        xyz -= np.median(xyz, axis=0)
        _u, _s, vh = np.linalg.svd(xyz, full_matrices=False)
        projected = xyz @ vh[:2].T
        extents = np.quantile(projected, 0.95, axis=0) - np.quantile(projected, 0.05, axis=0)
        long_extent, short_extent = sorted(map(float, extents), reverse=True)
        if short_extent <= 1e-8:
            continue
        aspect_error = abs((long_extent / short_extent) / (width_m / height_m) - 1.0)
        if aspect_error > 0.30:
            continue
        factor = float(np.median([width_m / long_extent, height_m / short_extent]))
        estimates.append({
            "capture": "before", "frame": item.get("frame"), "object_index": index,
            "point_count": int(len(xyz)), "extent_map_units": [long_extent, short_extent],
            "aspect_relative_error": aspect_error, "scale_m_per_map_unit": factor,
        })
    if len(estimates) < 2:
        return {
            "schema_version": 1, "status": "uncalibrated", "scale_m_per_map_unit": None,
            "reason": f"need at least two reliable {label} views; found {len(estimates)}",
            "reference_dimensions_m": {"width": width_m, "height": height_m},
            "views": estimates,
        }
    factors = np.asarray([view["scale_m_per_map_unit"] for view in estimates], dtype=np.float64)
    scale = float(np.median(factors))
    relative_mad = float(np.median(np.abs(factors - scale)) / scale)
    status = "provisional" if relative_mad <= 0.15 else "inconsistent"
    return {
        "schema_version": 1, "status": status,
        "coordinate_frame": report.get("coordinate_frame", "shared_colmap_map"),
        "reference_object": {"label": label, "dimensions_m": {"width": width_m, "height": height_m}},
        "scale_m_per_map_unit": scale if status == "provisional" else None,
        "relative_mad": relative_mad,
        "recommended_uncertainty_fraction": 0.15,
        "method": "median scale from robust PCA 5th-to-95th percentile extents of calibrated-depth mask points",
        "views": estimates,
        "warning": "This scale belongs only to this shared COLMAP reconstruction; rebuild and recalibrate for each new map.",
    }
