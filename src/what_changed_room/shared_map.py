"""Localize a second capture into a COLMAP reference map and make a browser preview."""

from __future__ import annotations

import html
import shutil
import sqlite3
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np


@dataclass(frozen=True)
class CameraView:
    name: str
    center: tuple[float, float, float]


def _read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated COLMAP binary model")
    return data


def read_sparse_geometry(model_dir: str | Path) -> tuple[np.ndarray, list[CameraView]]:
    """Read 3D points and camera centers from a COLMAP 3.x binary model."""
    model_dir = Path(model_dir)
    points_path = model_dir / "points3D.bin"
    images_path = model_dir / "images.bin"
    if not points_path.is_file() or not images_path.is_file():
        raise FileNotFoundError(f"expected images.bin and points3D.bin in {model_dir}")

    points = []
    with points_path.open("rb") as stream:
        (count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            point_id, x, y, z, _r, _g, _b, _error, track_len = struct.unpack(
                "<QdddBBBdQ", _read_exact(stream, struct.calcsize("<QdddBBBdQ"))
            )
            _read_exact(stream, int(track_len) * 8)
            points.append((x, y, z))

    cameras = []
    with images_path.open("rb") as stream:
        (count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            _image_id, = struct.unpack("<i", _read_exact(stream, 4))
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", _read_exact(stream, 56))
            _camera_id, = struct.unpack("<i", _read_exact(stream, 4))
            name = bytearray()
            while (char := _read_exact(stream, 1)) != b"\0":
                name.extend(char)
            (num_points2d,) = struct.unpack("<Q", _read_exact(stream, 8))
            _read_exact(stream, int(num_points2d) * 24)

            rotation = np.array([
                [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
                [2 * (qx*qy + qw*qz), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qw*qx)],
                [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx*qx + qy*qy)],
            ])
            center = -rotation.T @ np.array([tx, ty, tz])
            cameras.append(CameraView(name.decode("utf-8"), tuple(map(float, center))))

    return np.asarray(points, dtype=np.float64).reshape(-1, 3), cameras


def _select_reference_model(model_path: Path, database_names: set[str]) -> Path:
    """Accept a model folder or choose the best numbered model under sparse/."""
    if (model_path / "images.bin").is_file() and (model_path / "points3D.bin").is_file():
        return model_path
    candidates = [
        child for child in model_path.iterdir()
        if child.is_dir() and (child / "images.bin").is_file() and (child / "points3D.bin").is_file()
    ] if model_path.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"no binary COLMAP model found at {model_path} or in its immediate child folders"
        )

    def score(candidate: Path) -> tuple[int, int]:
        points, cameras = read_sparse_geometry(candidate)
        registered_in_database = sum(camera.name in database_names for camera in cameras)
        return registered_in_database, len(points)

    return max(candidates, key=score)


def _database_image_names(database_path: Path) -> list[str]:
    try:
        with sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True) as db:
            return [row[0] for row in db.execute("SELECT name FROM images ORDER BY image_id")]
    except sqlite3.Error as exc:
        raise ValueError(f"could not read COLMAP database {database_path}: {exc}") from exc


def _copy_database(source: Path, destination: Path) -> None:
    """Use SQLite backup so a source database with WAL files is copied consistently."""
    try:
        with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as src:
            with sqlite3.connect(destination) as dst:
                src.backup(dst)
    except sqlite3.Error as exc:
        raise ValueError(f"could not copy COLMAP database {source}: {exc}") from exc


def _prepare_after_images(after_dir: Path, root: Path) -> dict[str, Path]:
    """Copy only new frames into a clean folder; reference descriptors stay in the DB."""
    after_root = root / "after_images"
    after_root.mkdir(parents=True)
    after_lookup: dict[str, Path] = {}
    for index, source in enumerate(sorted(p for p in after_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})):
        # A distinct filename prevents collisions with reference images in the database.
        name = f"wcr_after_{index:05d}_{source.name}"
        destination = after_root / name
        shutil.copy2(source, destination)
        after_lookup[name] = source.resolve()
    if not after_lookup:
        raise ValueError(f"no JPG/PNG frames found under {after_dir}")
    return after_lookup


def _write_preview(
    output_html: Path,
    model_dir: Path,
    before_names: set[str],
    after_sources: dict[str, Path],
) -> dict[str, int]:
    points, cameras = read_sparse_geometry(model_dir)
    before_cameras = [camera for camera in cameras if camera.name in before_names]
    after_cameras = [camera for camera in cameras if camera.name in after_sources]

    preview_dir = output_html.parent / "localized_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, camera in enumerate(after_cameras):
        source = after_sources[camera.name]
        target = preview_dir / f"after_{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        cards.append(
            f'<figure><img src="localized_frames/{quote(target.name)}" alt="{html.escape(camera.name)}">'
            f'<figcaption>{html.escape(camera.name)}<br>camera center: '
            f'{camera.center[0]:.3f}, {camera.center[1]:.3f}, {camera.center[2]:.3f}</figcaption></figure>'
        )
    if not after_cameras:
        cards.append(
            '<p>No after frames localized. The red camera markers are absent because COLMAP did not '
            'find enough reliable matches. The map below is the reference/before map.</p>'
        )
        for index, source in enumerate(after_sources.values()):
            target = preview_dir / f"unlocalized_{index:03d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            cards.append(
                f'<figure><img src="localized_frames/{quote(target.name)}" alt="unlocalized after frame">'
                '<figcaption>Not localized in the shared map</figcaption></figure>'
            )

    all_xyz = [points]
    if cameras:
        all_xyz.append(np.asarray([c.center for c in cameras], dtype=np.float64))
    xyz = np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((1, 3))
    # Isometric projection: useful for a quick map inspection without extra plotting packages.
    projected = np.column_stack((xyz[:, 0] - xyz[:, 2], xyz[:, 1] - 0.5 * (xyz[:, 0] + xyz[:, 2])))
    low = projected.min(axis=0)
    span = np.maximum(projected.max(axis=0) - low, 1e-9)

    def xy(point: tuple[float, float, float]) -> tuple[float, float]:
        p = np.array([point[0] - point[2], point[1] - 0.5 * (point[0] + point[2])])
        x = 45 + 710 * (p[0] - low[0]) / span[0]
        y = 455 - 410 * (p[1] - low[1]) / span[1]
        return float(x), float(y)

    point_svg = "\n".join(
        f'<circle class="point" cx="{xy(tuple(point))[0]:.1f}" cy="{xy(tuple(point))[1]:.1f}" r="2"/>'
        for point in points
    )
    camera_svg = []
    for kind, values in (("before", before_cameras), ("after", after_cameras)):
        for camera in values:
            x, y = xy(camera.center)
            camera_svg.append(
                f'<circle class="camera {kind}" cx="{x:.1f}" cy="{y:.1f}" r="7"/>'
                f'<text x="{x + 9:.1f}" y="{y - 7:.1f}">{html.escape(Path(camera.name).name)}</text>'
            )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Shared COLMAP map preview</title>
<style>body{{font:16px system-ui;margin:24px;color:#202124}}svg{{width:min(100%,900px);height:auto;background:#f7f8fa;border:1px solid #ddd}}
.point{{fill:#888}}.camera.before{{fill:#2774c8}}.camera.after{{fill:#d44832}}text{{font:12px system-ui}}.legend{{display:flex;gap:24px}}figure{{width:180px;margin:0}}img{{width:180px;height:120px;object-fit:cover}}figcaption{{font-size:12px;overflow-wrap:anywhere}}.frames{{display:flex;flex-wrap:wrap;gap:12px}}</style></head>
<body><h1>Shared sparse map</h1>
<h2>How to read this page</h2>
<p>Each gray dot is a visual feature COLMAP matched across pictures; it is not a furniture object or a wall surface. Blue circles are the {len(before_cameras)} camera positions COLMAP reconstructed from the before video. Red circles would be after-video camera positions in the same coordinates; none registered in this run.</p>
<p>This view is a simplified projection of a sparse feature map, not a realistic room drawing. COLMAP distances here are in arbitrary units, not meters.</p>
<svg viewBox="0 0 800 500" role="img" aria-label="Isometric sparse map and camera positions"><g>{point_svg}</g><g>{''.join(camera_svg)}</g></svg>
<p class="legend"><span style="color:#2774c8">● Before cameras ({len(before_cameras)})</span><span style="color:#d44832">● After cameras ({len(after_cameras)})</span><span style="color:#888">● Sparse points ({len(points)})</span></p>
<h2>After-video frames</h2><p>These are the sampled after images. “Not localized” means COLMAP could not place that camera in the before map; it does not mean the scene had no changes.</p><div class="frames">{''.join(cards)}</div></body></html>"""
    output_html.write_text(document, encoding="utf-8")
    return {"points": len(points), "before_registered": len(before_cameras), "after_registered": len(after_cameras)}


def localize_after_images(
    before_model: str | Path,
    before_database: str | Path,
    after_images: str | Path,
    output_dir: str | Path,
    *,
    colmap_executable: str = "colmap",
) -> dict[str, object]:
    """Register after images to a before map, then write an HTML map/photo preview."""
    before_model = Path(before_model).resolve()
    database_path = Path(before_database).resolve()
    after_dir = Path(after_images).resolve()
    output_dir = Path(output_dir).resolve()
    before_names = set(_database_image_names(database_path))
    before_model = _select_reference_model(before_model, before_names)
    required = (before_model / "images.bin", before_model / "points3D.bin")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"reference model must contain images.bin and points3D.bin: {before_model}")
    if not after_dir.is_dir():
        raise NotADirectoryError(f"after image path must be a directory: {after_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_model = output_dir / "shared_model"
    if result_model.exists():
        raise FileExistsError(
            f"output model already exists: {result_model}; choose a new --output-dir "
            "or remove this generated output explicitly before rerunning"
        )
    # Keep the temporary database under the project output path. Snap-confined
    # COLMAP builds may have a private /tmp and fail to see Python's temp files.
    with tempfile.TemporaryDirectory(prefix="wcr-localize-", dir=output_dir) as temporary:
        work = Path(temporary)
        database = work / "database.db"
        _copy_database(database_path, database)
        after_sources = _prepare_after_images(after_dir, work)
        image_root = work / "after_images"
        commands = [
            [colmap_executable, "feature_extractor", "--database_path", str(database), "--image_path", str(image_root), "--SiftExtraction.use_gpu", "0"],
            [colmap_executable, "exhaustive_matcher", "--database_path", str(database), "--SiftMatching.use_gpu", "0"],
            [colmap_executable, "image_registrator", "--database_path", str(database), "--input_path", str(before_model), "--output_path", str(result_model)],
        ]
        registration_error = None
        for index, command in enumerate(commands):
            if index == len(commands) - 1:
                # COLMAP's image_registrator requires the output directory to exist.
                result_model.mkdir()
            print("$ " + " ".join(command), flush=True)
            try:
                completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError as exc:
                raise RuntimeError(f"could not start COLMAP executable {colmap_executable!r}: {exc}") from exc
            unknown_option = "unrecognised option" in completed.stdout.lower() or "unrecognized option" in completed.stdout.lower()
            if completed.returncode and unknown_option and index in (0, 1):
                # COLMAP 3.7 names these SIFT options directly; newer versions expose
                # the GPU switch on the feature-extraction/matching option groups.
                fallback = list(command)
                fallback_flag = "--FeatureExtraction.use_gpu" if index == 0 else "--FeatureMatching.use_gpu"
                fallback[fallback.index(next(value for value in fallback if value.startswith("--") and ".use_gpu" in value))] = fallback_flag
                print("Retrying with this COLMAP version's option name:", " ".join(fallback), flush=True)
                completed = subprocess.run(fallback, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if completed.returncode:
                tail = completed.stdout[-6000:]
                if index == len(commands) - 1:
                    registration_error = tail or f"COLMAP exited with status {completed.returncode}"
                    break
                raise RuntimeError(f"COLMAP command failed ({completed.returncode}): {' '.join(command)}\n{tail}")

        preview_model = result_model
        if registration_error or not (preview_model / "images.bin").is_file() or not (preview_model / "points3D.bin").is_file():
            if result_model.exists():
                shutil.rmtree(result_model)
            preview_model = before_model
        counts = _write_preview(output_dir / "shared_map.html", preview_model, before_names, after_sources)

    return {
        **counts,
        "localization_success": counts["after_registered"] > 0,
        "failure_reason": registration_error,
        "shared_model": str(result_model) if preview_model == result_model else None,
        "reference_model": str(before_model),
        "preview_model": str(preview_model),
        "preview": str(output_dir / "shared_map.html"),
    }
