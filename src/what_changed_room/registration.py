"""Scale-aware registration for sparse point clouds and matched 3D features.

Transforms map source/after coordinates into target/before coordinates. Cross-
capture correspondences must be created from image evidence; point IDs from two
independent COLMAP models are not shared identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .reconstruction import PointCloud

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SimilarityTransform:
    """Map points from ``source_frame`` to ``target_frame`` as ``s * R * p + t``."""

    scale: float
    rotation: FloatArray
    translation: FloatArray
    source_frame: str = "after"
    target_frame: str = "before"
    scale_status: str = "estimated"

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be a finite positive value")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation must be a finite 3-vector")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: FloatArray) -> FloatArray:
        """Transform an ``(N, 3)`` point array into the target frame."""
        points = _points(points, "points")
        return self.scale * (points @ self.rotation.T) + self.translation

    @property
    def matrix(self) -> FloatArray:
        """Return a homogeneous matrix; its upper-left block includes scale."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class RegistrationDiagnostics:
    """Measured registration quality; success means configured gates passed."""

    success: bool
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    source_coverage: float
    target_coverage: float
    rmse: float | None
    median_error: float | None
    ransac_iterations: int
    icp_iterations: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class RegistrationResult:
    """Estimated source-to-target transform and its diagnostics."""

    transform: SimilarityTransform | None
    diagnostics: RegistrationDiagnostics


def _points(value: FloatArray, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def _non_collinear(points: FloatArray) -> bool:
    edges = points[1:] - points[0]
    if len(edges) < 2:
        return False
    area = np.linalg.norm(np.cross(edges[0], edges[1]))
    scale = max(float(np.linalg.norm(edges, axis=1).max()), 1.0)
    return bool(area > np.finfo(np.float64).eps * scale * scale * 100)


def _fit_similarity(source: FloatArray, target: FloatArray, estimate_scale: bool) -> SimilarityTransform:
    """Least-squares Umeyama fit mapping paired source points to target points."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.sum(source_centered * source_centered) / len(source))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("correspondences have no spatial spread")

    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance) if estimate_scale else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("correspondences imply an invalid scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def _compose(left: SimilarityTransform, right: SimilarityTransform) -> SimilarityTransform:
    """Return ``left(right(points))``."""
    rotation = left.rotation @ right.rotation
    scale = left.scale * right.scale
    translation = left.scale * (left.rotation @ right.translation) + left.translation
    return SimilarityTransform(
        scale,
        rotation,
        translation,
        right.source_frame,
        left.target_frame,
        right.scale_status if left.scale == 1.0 else left.scale_status,
    )


def refine_icp(
    source_points: FloatArray,
    target_points: FloatArray,
    initial: SimilarityTransform,
    *,
    max_correspondence_distance: float,
    max_iterations: int = 20,
    trim_fraction: float = 0.8,
    tolerance: float = 1e-6,
) -> tuple[SimilarityTransform, int, float | None]:
    """Refine a good initialization with trimmed point-to-point rigid ICP.

    The estimated scale is held fixed. Nearest-neighbor search uses a KD-tree;
    points farther than the gate are ignored and the worst residual fraction is
    trimmed to reduce the effect of changed objects and sparse outliers.
    """
    source = _points(source_points, "source_points")
    target = _points(target_points, "target_points")
    if max_correspondence_distance <= 0 or not np.isfinite(max_correspondence_distance):
        raise ValueError("max_correspondence_distance must be finite and positive")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0, 1]")

    tree = cKDTree(target)
    transform = initial
    previous_rmse = np.inf
    used_iterations = 0
    for _ in range(max_iterations):
        transformed = transform.apply(source)
        distances, indices = tree.query(transformed, k=1, distance_upper_bound=max_correspondence_distance)
        valid = np.isfinite(distances) & (indices < len(target))
        if int(valid.sum()) < 3:
            break
        valid_indices = np.flatnonzero(valid)
        keep_count = max(3, int(np.ceil(len(valid_indices) * trim_fraction)))
        keep = valid_indices[np.argpartition(distances[valid], keep_count - 1)[:keep_count]]
        src_matches = transformed[keep]
        dst_matches = target[indices[keep]]
        if not _non_collinear(src_matches):
            break
        delta = _fit_similarity(src_matches, dst_matches, estimate_scale=False)
        delta = SimilarityTransform(1.0, delta.rotation, delta.translation,
                                    transform.target_frame, transform.target_frame,
                                    transform.scale_status)
        transform = _compose(delta, transform)
        residual = np.linalg.norm(transform.apply(source[keep]) - dst_matches, axis=1)
        rmse = float(np.sqrt(np.mean(residual * residual)))
        used_iterations += 1
        if abs(previous_rmse - rmse) <= tolerance:
            previous_rmse = rmse
            break
        previous_rmse = rmse
    return transform, used_iterations, None if not np.isfinite(previous_rmse) else previous_rmse


def register_point_clouds(
    source: PointCloud,
    target: PointCloud,
    correspondences: IntArray,
    *,
    source_frame: str = "after",
    target_frame: str = "before",
    scale_known: bool = False,
    inlier_threshold: float = 0.03,
    max_iterations: int = 1000,
    min_inliers: int = 4,
    min_inlier_ratio: float = 0.25,
    random_seed: int = 0,
    icp_iterations: int = 20,
    icp_max_distance: float | None = None,
) -> RegistrationResult:
    """Estimate a robust transform using matched 3D points, then refine by ICP.

    ``correspondences[:, 0]`` indexes the source/after cloud and column 1
    indexes the target/before cloud. Similarity scale is estimated unless
    ``scale_known`` is true. Weak matches return a failed result with no
    transform rather than silently aligning the scenes.
    """
    src, dst = source.points, target.points
    pairs = np.asarray(correspondences, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("correspondences must have shape (K, 2): source index, target index")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs[:, 0] >= len(src)) or np.any(pairs[:, 1] >= len(dst))):
        raise ValueError("correspondence index is outside its point cloud")
    if not np.isfinite(inlier_threshold) or inlier_threshold <= 0:
        raise ValueError("inlier_threshold must be finite and positive")
    if max_iterations <= 0 or min_inliers < 3 or not 0 < min_inlier_ratio <= 1:
        raise ValueError("max_iterations must be positive, min_inliers at least 3, and ratio in (0, 1]")

    source_matches = src[pairs[:, 0]] if len(pairs) else np.empty((0, 3))
    target_matches = dst[pairs[:, 1]] if len(pairs) else np.empty((0, 3))
    pair_count = len(pairs)

    def failed(
        reason: str,
        iterations: int = 0,
        mask: NDArray[np.bool_] | None = None,
        errors: FloatArray | None = None,
    ) -> RegistrationResult:
        count = int(mask.sum()) if mask is not None else 0
        ratio = count / pair_count if pair_count else 0.0
        rmse = float(np.sqrt(np.mean(errors[mask] ** 2))) if count and errors is not None else None
        median = float(np.median(errors[mask])) if count and errors is not None else None
        source_coverage = len(np.unique(pairs[mask, 0])) / len(src) if count and mask is not None else 0.0
        target_coverage = len(np.unique(pairs[mask, 1])) / len(dst) if count and mask is not None else 0.0
        diagnostics = RegistrationDiagnostics(
            False, pair_count, count, ratio, source_coverage, target_coverage,
            rmse, median, iterations, 0, reason,
        )
        return RegistrationResult(None, diagnostics)

    if pair_count < max(3, min_inliers):
        return failed(f"need at least {max(3, min_inliers)} matched 3D points; got {pair_count}")

    rng = np.random.default_rng(random_seed)
    best_mask: NDArray[np.bool_] | None = None
    best_candidate: SimilarityTransform | None = None
    best_count = 0
    best_median = np.inf
    ran = 0
    estimate_scale = not scale_known
    for ran in range(1, max_iterations + 1):
        sample = rng.choice(pair_count, size=3, replace=False)
        if (not _non_collinear(source_matches[sample])
                or not _non_collinear(target_matches[sample])):
            continue
        try:
            candidate = _fit_similarity(source_matches[sample], target_matches[sample], estimate_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = np.linalg.norm(candidate.apply(source_matches) - target_matches, axis=1)
        mask = errors <= inlier_threshold
        count = int(mask.sum())
        median = float(np.median(errors[mask])) if count else np.inf
        if count > best_count or (count == best_count and median < best_median):
            best_mask, best_candidate = mask, candidate
            best_count, best_median = count, median
            ratio = count / pair_count
            if ratio >= 1:
                break
            if ratio > 0:
                success_probability = ratio ** 3
                if success_probability < 1:
                    required = int(np.ceil(np.log(1 - 0.999) / np.log(1 - success_probability)))
                    if ran >= max(50, required):
                        break

    if best_mask is None or best_count < min_inliers:
        best_errors = (np.linalg.norm(best_candidate.apply(source_matches) - target_matches, axis=1)
                       if best_candidate is not None else None)
        return failed(f"no robust alignment found with {min_inliers} inliers required", ran,
                      best_mask, best_errors)

    try:
        coarse = _fit_similarity(source_matches[best_mask], target_matches[best_mask], estimate_scale)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return failed(f"inlier refinement failed: {exc}", ran)
    coarse = SimilarityTransform(coarse.scale, coarse.rotation, coarse.translation,
                                 source_frame, target_frame,
                                 "known" if scale_known else "estimated")

    final = coarse
    used_icp = 0
    if icp_iterations:
        gate = icp_max_distance if icp_max_distance is not None else inlier_threshold * 2.0
        refined, used_icp, _ = refine_icp(src, dst, coarse,
                                          max_correspondence_distance=gate,
                                          max_iterations=icp_iterations)
        # Keep ICP only if it preserves the cross-capture feature consensus.
        old_error = np.linalg.norm(coarse.apply(source_matches) - target_matches, axis=1)
        new_error = np.linalg.norm(refined.apply(source_matches) - target_matches, axis=1)
        old_rmse = float(np.sqrt(np.mean(old_error[best_mask] ** 2)))
        new_rmse = float(np.sqrt(np.mean(new_error[best_mask] ** 2)))
        if new_rmse <= old_rmse + inlier_threshold * 0.05:
            final = refined

    errors = np.linalg.norm(final.apply(source_matches) - target_matches, axis=1)
    inlier_mask = errors <= inlier_threshold
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / pair_count
    rmse = float(np.sqrt(np.mean(errors[inlier_mask] ** 2))) if inlier_count else None
    median = float(np.median(errors[inlier_mask])) if inlier_count else None
    source_coverage = len(np.unique(pairs[inlier_mask, 0])) / len(src) if inlier_count else 0.0
    target_coverage = len(np.unique(pairs[inlier_mask, 1])) / len(dst) if inlier_count else 0.0
    success = inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio
    reason = None if success else (
        f"alignment failed quality gates: {inlier_count} inliers, ratio {inlier_ratio:.3f}"
    )
    diagnostics = RegistrationDiagnostics(
        success, pair_count, inlier_count, inlier_ratio, source_coverage, target_coverage,
        rmse, median, ran, used_icp, reason,
    )
    return RegistrationResult(final if success else None, diagnostics)
