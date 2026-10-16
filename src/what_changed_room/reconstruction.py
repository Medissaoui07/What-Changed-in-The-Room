"""Replaceable reconstruction backends and lightweight point-cloud importers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PointCloudFormatError(ValueError):
    """Raised for malformed or unsupported point-cloud files."""


@dataclass(frozen=True)
class PointCloud:
    """3D points and optional RGB colors normalized to [0, 1]."""

    points: NDArray[np.float64]
    colors: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("points must be a non-empty array with shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("points must contain only finite coordinates")
        object.__setattr__(self, "points", points)
        if self.colors is not None:
            colors = np.asarray(self.colors, dtype=np.float64)
            if colors.shape != points.shape or not np.isfinite(colors).all():
                raise ValueError("colors must be a finite array matching points with shape (N, 3)")
            if colors.max(initial=0) > 1:
                colors = colors / 255.0
            object.__setattr__(self, "colors", np.clip(colors, 0, 1))


@dataclass(frozen=True)
class ReconstructionResult:
    """Backend output; camera poses, if present, are camera-to-world 4x4 matrices."""

    point_cloud: PointCloud
    coordinate_frame: str = "source"
    scale_status: str = "unknown"
    camera_poses: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReconstructionBackend(ABC):
    """Contract for reconstructing one scene from extracted image frames."""

    @abstractmethod
    def reconstruct(self, frames: list[Path], output_dir: Path) -> ReconstructionResult:
        """Reconstruct geometry and return it with coordinate-frame provenance."""


def _ply(path: Path) -> PointCloud:
    ply_types = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
                 "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
                 "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
                 "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise PointCloudFormatError(f"Not a PLY file: {path}")
        fmt, count, props, in_vertex = None, None, [], False
        while True:
            line = stream.readline()
            if not line:
                raise PointCloudFormatError("PLY header has no end_header")
            parts = line.decode("ascii").strip().split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if len(parts) != 3:
                    raise PointCloudFormatError("PLY vertex list properties are unsupported")
                props.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break
        names = [name for name, _ in props]
        if not count or not all(axis in names for axis in ("x", "y", "z")):
            raise PointCloudFormatError("PLY must declare vertices with x, y, and z")
        if fmt == "ascii":
            try:
                rows = np.loadtxt(stream, dtype=np.float64, max_rows=count, ndmin=2)
            except ValueError as exc:
                raise PointCloudFormatError(f"Invalid ASCII PLY data: {exc}") from exc
            if rows.shape != (count, len(props)):
                raise PointCloudFormatError("PLY vertex count does not match data")
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt == "binary_little_endian" else ">"
            try:
                dtype = np.dtype([(n, endian + ply_types[t]) for n, t in props])
            except KeyError as exc:
                raise PointCloudFormatError(f"Unsupported PLY scalar type: {exc.args[0]}") from exc
            records = np.fromfile(stream, dtype=dtype, count=count)
            if len(records) != count:
                raise PointCloudFormatError("PLY vertex count does not match binary data")
            rows = np.column_stack([records[n] for n in names]).astype(np.float64)
        else:
            raise PointCloudFormatError(f"Unsupported PLY encoding: {fmt}")
        xyz = rows[:, [names.index(a) for a in ("x", "y", "z")]]
        rgb = next((cs for cs in (("red", "green", "blue"), ("r", "g", "b")) if all(c in names for c in cs)), None)
        colors = rows[:, [names.index(c) for c in rgb]] if rgb else None
        return PointCloud(xyz, colors)


_PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "i1", ("I", 2): "<i2",
              ("I", 4): "<i4", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}


def _pcd(path: Path) -> PointCloud:
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise PointCloudFormatError("PCD header has no DATA declaration")
            parts = line.decode("ascii", errors="replace").strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                encoding = parts[1].lower() if len(parts) > 1 else ""
                break
        fields = header.get("FIELDS", header.get("FIELD", []))
        sizes = list(map(int, header.get("SIZE", [])))
        types = header.get("TYPE", [])
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        n = int(header.get("POINTS", [str(int(header.get("WIDTH", ["0"])[0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
        if n <= 0 or not all(a in fields for a in ("x", "y", "z")):
            raise PointCloudFormatError("PCD must contain points and x, y, z fields")
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise PointCloudFormatError("PCD FIELDS, SIZE, TYPE, and COUNT declarations must agree")
        if encoding == "ascii":
            try:
                rows = np.loadtxt(stream, dtype=np.float64, max_rows=n, ndmin=2)
            except ValueError as exc:
                raise PointCloudFormatError(f"Invalid ASCII PCD data: {exc}") from exc
            if rows.shape != (n, sum(counts)):
                raise PointCloudFormatError("PCD point count does not match ASCII data")
            offsets = np.cumsum([0] + counts[:-1]).astype(int)
            coords = rows[:, [offsets[fields.index(a)] for a in ("x", "y", "z")]]
            color_fields = next((cs for cs in (("r", "g", "b"), ("red", "green", "blue")) if all(a in fields for a in cs)), None)
            colors = rows[:, [offsets[fields.index(a)] for a in color_fields]] if color_fields else None
        elif encoding == "binary":
            try:
                dtype = np.dtype([(f, _PCD_TYPES[(t, s)]) if c == 1 else (f, _PCD_TYPES[(t, s)], (c,))
                                  for f, s, t, c in zip(fields, sizes, types, counts)])
            except KeyError as exc:
                raise PointCloudFormatError(f"Unsupported PCD field type: {exc.args[0]}") from exc
            records = np.fromfile(stream, dtype=dtype, count=n)
            if len(records) != n:
                raise PointCloudFormatError("PCD point count does not match binary data")
            coords = np.column_stack([records[a] for a in ("x", "y", "z")]).astype(np.float64)
            color_fields = next((cs for cs in (("r", "g", "b"), ("red", "green", "blue")) if all(a in fields for a in cs)), None)
            colors = np.column_stack([records[a] for a in color_fields]).astype(np.float64) if color_fields else None
        else:
            raise PointCloudFormatError(f"Unsupported PCD DATA encoding '{encoding}'; use ascii or binary")
        return PointCloud(coords, colors)


def load_point_cloud(path: str | Path) -> PointCloud:
    """Load ASCII/binary PLY or ASCII/uncompressed-binary PCD geometry."""
    source = Path(path)
    if source.suffix.lower() not in (".ply", ".pcd"):
        raise PointCloudFormatError(f"Unsupported point-cloud extension '{source.suffix}'; expected .ply or .pcd")
    if not source.is_file():
        raise FileNotFoundError(f"Point-cloud file does not exist: {source}")
    if source.suffix.lower() == ".ply":
        return _ply(source)
    if source.suffix.lower() == ".pcd":
        return _pcd(source)
    raise AssertionError("suffix was validated above")


def import_point_cloud(path: str | Path) -> ReconstructionResult:
    """Load an existing point cloud through the shared reconstruction result interface."""
    source = Path(path)
    return ReconstructionResult(load_point_cloud(source), f"source:{source.name}", "unknown",
                               metadata={"source_path": str(source.resolve()), "format": source.suffix.lower()[1:]})
