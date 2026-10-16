"""Relative monocular depth and conservative calibration to COLMAP camera Z."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class RelativeDepthEstimator(Protocol):
    """Model interface returning a full-resolution relative-depth array."""

    def predict(self, image_path: Path) -> np.ndarray:
        """Predict relative depth; units and scale are model-dependent."""


class DepthAnythingV2SmallEstimator:
    """Lazy Hugging Face adapter for the official Depth Anything V2 Small model."""

    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, *, device: str = "auto", model_path: str | Path | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything inference requires the optional 'depth' dependencies"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = f"cuda:{device}"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch environment cannot access a CUDA device"
            )
        self.torch = torch
        self.device = device
        model_source: str | Path = self.MODEL_ID if model_path is None else model_path
        is_local_path = isinstance(model_source, Path) or (
            isinstance(model_source, str) and model_source.startswith(("~", "/", "."))
        )
        if is_local_path:
            local_dir = Path(model_source).expanduser().resolve()
            required_files = ("config.json", "preprocessor_config.json", "model.safetensors")
            missing = [name for name in required_files if not (local_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"local Depth Anything model directory {local_dir} is missing {', '.join(missing)}; "
                    "download the model's config.json and preprocessor_config.json beside model.safetensors"
                )
            model_source = str(local_dir)
            local_files_only = True
        else:
            local_files_only = False
        self.model_source = str(model_source)
        self.processor = AutoImageProcessor.from_pretrained(
            model_source, local_files_only=local_files_only
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_source, local_files_only=local_files_only
        )
        self.model.to(device).eval()

    def predict(self, image_path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"could not read RGB frame for depth: {image_path}")
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            full_resolution = self.torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        prediction = full_resolution.float().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(prediction).all():
            raise ValueError(f"depth model produced non-finite values for {image_path}")
        return prediction


@dataclass(frozen=True)
class DepthAnchor:
    """One COLMAP 2D feature and its positive camera-space Z value."""

    x: float
    y: float
    camera_z: float
    point3d_id: int


@dataclass(frozen=True)
class DepthCalibration:
    """Fit and held-out quality diagnostics for relative depth to camera Z."""

    status: str
    scale: float | None
    offset: float | None
    anchor_count: int
    train_count: int
    train_inlier_count: int
    train_inlier_ratio: float | None
    coverage_cells: int
    coverage_fraction: float
    holdout_count: int
    holdout_median_relative_error: float | None
    holdout_p90_relative_error: float | None
    positive_pixel_fraction: float | None
    failure_reasons: tuple[str, ...]


def _read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated COLMAP binary model")
    return data


def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert COLMAP's Hamilton quaternion to a world-to-camera rotation."""
    return np.asarray([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
        [2 * (qx*qy + qw*qz), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qw*qx)],
        [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx*qx + qy*qy)],
    ], dtype=np.float64)


def read_colmap_depth_anchors(model_dir: str | Path) -> dict[str, list[DepthAnchor]]:
    """Read registered feature observations and express their map points in camera Z.

    Returned image keys are the names stored in the COLMAP model. Camera Z does
    not require intrinsics because COLMAP stores the world-to-camera pose and
    every feature's measured source-image pixel coordinate.
    """
    model_dir = Path(model_dir)
    images_path, points_path = model_dir / "images.bin", model_dir / "points3D.bin"
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError(f"expected images.bin and points3D.bin in {model_dir}")

    point_xyz: dict[int, np.ndarray] = {}
    with points_path.open("rb") as stream:
        (count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(count):
            point_id, x, y, z, _red, _green, _blue, _error, track_length = struct.unpack(
                "<QdddBBBdQ", _read_exact(stream, struct.calcsize("<QdddBBBdQ"))
            )
            _read_exact(stream, int(track_length) * 8)
            point_xyz[int(point_id)] = np.asarray([x, y, z], dtype=np.float64)

    image_anchors: dict[str, list[DepthAnchor]] = {}
    with images_path.open("rb") as stream:
        (image_count,) = struct.unpack("<Q", _read_exact(stream, 8))
        for _ in range(image_count):
            _image_id, = struct.unpack("<i", _read_exact(stream, 4))
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", _read_exact(stream, 56))
            _camera_id, = struct.unpack("<i", _read_exact(stream, 4))
            name_bytes = bytearray()
            while (char := _read_exact(stream, 1)) != b"\0":
                name_bytes.extend(char)
            (observation_count,) = struct.unpack("<Q", _read_exact(stream, 8))
            rotation = _quaternion_rotation(qw, qx, qy, qz)
            translation = np.asarray([tx, ty, tz], dtype=np.float64)
            anchors: list[DepthAnchor] = []
            for _ in range(observation_count):
                x, y, point_id = struct.unpack("<ddq", _read_exact(stream, 24))
                xyz = point_xyz.get(int(point_id))
                if xyz is None:
                    continue
                camera_z = float((rotation @ xyz + translation)[2])
                if math.isfinite(camera_z) and camera_z > 1e-6:
                    anchors.append(DepthAnchor(float(x), float(y), camera_z, int(point_id)))
            image_anchors[name_bytes.decode("utf-8")] = anchors
    return image_anchors


def anchors_for_frame(
    image_anchors: dict[str, list[DepthAnchor]], frame_name: str, capture: str
) -> tuple[str | None, list[DepthAnchor]]:
    """Find the registered COLMAP image corresponding to a source frame."""
    if capture == "before" and frame_name in image_anchors:
        return frame_name, image_anchors[frame_name]
    matches = [
        name for name in image_anchors
        if name == frame_name or name.endswith("_" + frame_name)
    ]
    if len(matches) != 1:
        return None, []
    return matches[0], image_anchors[matches[0]]


def calibrate_relative_depth(
    relative_depth: np.ndarray,
    anchors: list[DepthAnchor],
    *,
    minimum_anchors: int = 20,
    grid_size: int = 4,
    minimum_coverage_cells: int = 4,
    random_seed: int = 7,
) -> DepthCalibration:
    """Fit ``camera_z = scale * relative_depth + offset`` with robust holdout checks.

    The fit is accepted only with enough distributed training anchors and good
    held-out residuals. A rejected fit must not be used for 3D unprojection.
    """
    height, width = relative_depth.shape[:2]
    valid: list[tuple[float, float, float, int]] = []
    for anchor in anchors:
        x, y = int(round(anchor.x)), int(round(anchor.y))
        if 0 <= x < width and 0 <= y < height:
            value = float(relative_depth[y, x])
            if math.isfinite(value) and math.isfinite(anchor.camera_z):
                valid.append((value, anchor.camera_z, anchor.x, anchor.y))

    count = len(valid)
    base = dict(
        status="rejected", scale=None, offset=None, anchor_count=count,
        train_count=0, train_inlier_count=0, train_inlier_ratio=None,
        coverage_cells=0, coverage_fraction=0.0, holdout_count=0,
        holdout_median_relative_error=None, holdout_p90_relative_error=None,
        positive_pixel_fraction=None,
        failure_reasons=(),
    )
    if count < minimum_anchors:
        return DepthCalibration(**(base | {"failure_reasons": (f"only {count} valid sparse anchors; need {minimum_anchors}",)}))

    samples = np.asarray(valid, dtype=np.float64)
    random = np.random.default_rng(random_seed)
    order = random.permutation(count)
    holdout_count = max(5, int(round(count * 0.2)))
    if count - holdout_count < 10:
        holdout_count = count - 10
    holdout_ids, train_ids = order[:holdout_count], order[holdout_count:]
    if len(holdout_ids) < 5:
        return DepthCalibration(**(base | {"failure_reasons": ("not enough anchors for a held-out fit check",)}))

    train = samples[train_ids]
    depth_values, z_values = train[:, 0], train[:, 1]
    if float(np.ptp(depth_values)) <= 1e-8:
        return DepthCalibration(**(base | {
            "train_count": len(train), "holdout_count": len(holdout_ids),
            "failure_reasons": ("relative-depth anchors have no usable variation",),
        }))

    # A small deterministic RANSAC search limits sensitivity to bad sparse tracks.
    pair_count = min(512, max(128, len(train) * 8))
    best_mask: np.ndarray | None = None
    best_median = float("inf")
    z_scale = max(float(np.median(z_values)), 1e-6)
    residual_limit = max(0.03 * z_scale, 1e-6)
    for _ in range(pair_count):
        first, second = random.choice(len(train), size=2, replace=False)
        depth_delta = depth_values[second] - depth_values[first]
        if abs(float(depth_delta)) <= 1e-10:
            continue
        scale = (z_values[second] - z_values[first]) / depth_delta
        offset = z_values[first] - scale * depth_values[first]
        residuals = np.abs(scale * depth_values + offset - z_values)
        inliers = residuals <= residual_limit
        inlier_count = int(inliers.sum())
        median = float(np.median(residuals[inliers])) if inlier_count else float("inf")
        if best_mask is None or inlier_count > int(best_mask.sum()) or (
            inlier_count == int(best_mask.sum()) and median < best_median
        ):
            best_mask, best_median = inliers, median

    if best_mask is None or int(best_mask.sum()) < 2:
        return DepthCalibration(**(base | {
            "train_count": len(train), "holdout_count": len(holdout_ids),
            "failure_reasons": ("could not fit a stable relative-depth mapping",),
        }))
    design = np.column_stack((depth_values[best_mask], np.ones(int(best_mask.sum()))))
    scale, offset = np.linalg.lstsq(design, z_values[best_mask], rcond=None)[0]
    train_residuals = np.abs(scale * depth_values + offset - z_values)
    inliers = train_residuals <= residual_limit
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(train)

    xs, ys = train[inliers, 2], train[inliers, 3]
    grid_x = np.clip((xs / max(width, 1) * grid_size).astype(int), 0, grid_size - 1)
    grid_y = np.clip((ys / max(height, 1) * grid_size).astype(int), 0, grid_size - 1)
    cells = len(set(zip(grid_x.tolist(), grid_y.tolist(), strict=True)))
    coverage = cells / float(grid_size * grid_size)

    holdout = samples[holdout_ids]
    holdout_prediction = scale * holdout[:, 0] + offset
    relative_errors = np.abs(holdout_prediction - holdout[:, 1]) / np.maximum(holdout[:, 1], 1e-6)
    holdout_median = float(np.median(relative_errors))
    holdout_p90 = float(np.quantile(relative_errors, 0.9))
    projected_camera_z = scale * relative_depth + offset
    finite_pixels = np.isfinite(projected_camera_z)
    positive_pixel_fraction = (
        float(np.mean(projected_camera_z[finite_pixels] > 0)) if finite_pixels.any() else 0.0
    )

    failures: list[str] = []
    if inlier_count < 10 or inlier_ratio < 0.5:
        failures.append("too few robust training inliers")
    if cells < minimum_coverage_cells:
        failures.append("sparse anchors do not cover enough image regions")
    if holdout_median > 0.10 or holdout_p90 > 0.25:
        failures.append("held-out camera-Z error exceeds the 10% median / 25% p90 limits")
    if positive_pixel_fraction < 0.9:
        failures.append("affine fit predicts non-positive camera Z across more than 10% of the image")
    status = "calibrated" if not failures else "rejected"
    return DepthCalibration(
        status=status,
        scale=float(scale) if status == "calibrated" else None,
        offset=float(offset) if status == "calibrated" else None,
        anchor_count=count,
        train_count=len(train),
        train_inlier_count=inlier_count,
        train_inlier_ratio=float(inlier_ratio),
        coverage_cells=cells,
        coverage_fraction=float(coverage),
        holdout_count=len(holdout_ids),
        holdout_median_relative_error=holdout_median,
        holdout_p90_relative_error=holdout_p90,
        positive_pixel_fraction=positive_pixel_fraction,
        failure_reasons=tuple(failures),
    )


def write_depth_preview(depth: np.ndarray, output_path: str | Path) -> Path:
    """Write a percentile-normalized visualization without changing raw depth."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = depth[np.isfinite(depth)]
    if not finite.size:
        raise ValueError("cannot create a depth preview with no finite pixels")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    pixels = np.nan_to_num(normalized * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(pixels, cv2.COLORMAP_TURBO)
    if not cv2.imwrite(str(output_path), preview):
        raise OSError(f"could not write depth preview: {output_path}")
    return output_path


def write_depth_report(report: dict[str, object], output_path: str | Path) -> Path:
    """Atomically write depth inference and calibration diagnostics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def calibration_as_dict(calibration: DepthCalibration) -> dict[str, object]:
    """Convert calibration diagnostics to JSON-compatible values."""
    return asdict(calibration)
