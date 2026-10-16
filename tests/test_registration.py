import numpy as np

from what_changed_room.reconstruction import PointCloud
from what_changed_room.registration import (
    SimilarityTransform,
    refine_icp,
    register_point_clouds,
)


def _rotation_z(angle: float) -> np.ndarray:
    return np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


def test_ransac_recovers_scale_rotation_translation_with_outliers() -> None:
    rng = np.random.default_rng(17)
    source_points = rng.uniform(-2.0, 2.0, size=(120, 3))
    expected = SimilarityTransform(
        1.7, _rotation_z(0.37), np.array([1.2, -0.8, 0.35]),
    )
    target_points = expected.apply(source_points)
    target = PointCloud(target_points)

    good = np.column_stack((np.arange(90), np.arange(90)))
    outliers = np.column_stack((np.arange(90, 120), rng.integers(0, 90, size=30)))
    pairs = np.vstack((good, outliers))
    rng.shuffle(pairs)

    result = register_point_clouds(
        PointCloud(source_points), target, pairs,
        inlier_threshold=1e-5,
        max_iterations=1000,
        random_seed=9,
        icp_iterations=0,
    )

    assert result.diagnostics.success
    assert result.transform is not None
    assert result.diagnostics.inlier_count == 90
    assert result.transform.source_frame == "after"
    assert result.transform.target_frame == "before"
    assert result.transform.scale_status == "estimated"
    assert np.isclose(result.transform.scale, expected.scale, atol=1e-8)
    assert np.allclose(result.transform.rotation, expected.rotation, atol=1e-8)
    assert np.allclose(result.transform.translation, expected.translation, atol=1e-8)


def test_known_metric_scale_keeps_transform_rigid() -> None:
    rng = np.random.default_rng(23)
    source_points = rng.normal(size=(30, 3))
    expected = SimilarityTransform(1.0, _rotation_z(-0.2), np.array([-0.4, 0.2, 1.1]))
    target_points = expected.apply(source_points)
    pairs = np.column_stack((np.arange(len(source_points)), np.arange(len(source_points))))

    result = register_point_clouds(
        PointCloud(source_points), PointCloud(target_points), pairs,
        scale_known=True, inlier_threshold=1e-5, icp_iterations=0,
    )

    assert result.diagnostics.success
    assert result.transform is not None
    assert result.transform.scale_status == "known"
    assert result.transform.scale == 1.0
    assert np.allclose(result.transform.matrix[:3, :3], expected.rotation, atol=1e-8)
    assert np.allclose(result.transform.translation, expected.translation, atol=1e-8)


def test_weak_correspondences_fail_without_returning_transform() -> None:
    rng = np.random.default_rng(4)
    source_points = rng.normal(size=(40, 3))
    target_points = rng.normal(size=(40, 3)) + 10.0
    pairs = np.column_stack((np.arange(20), rng.permutation(np.arange(20))))

    result = register_point_clouds(
        PointCloud(source_points), PointCloud(target_points), pairs,
        inlier_threshold=1e-4, max_iterations=150, random_seed=3,
        icp_iterations=0,
    )

    assert not result.diagnostics.success
    assert result.transform is None
    assert result.diagnostics.failure_reason is not None


def test_icp_refines_translation_without_changing_estimated_scale() -> None:
    rng = np.random.default_rng(31)
    source_points = rng.uniform(-1.0, 1.0, size=(80, 3))
    rotation = _rotation_z(0.18)
    true_translation = np.array([0.2, -0.3, 0.1])
    target_points = 1.4 * (source_points @ rotation.T) + true_translation
    initial = SimilarityTransform(
        1.4, rotation, true_translation + np.array([0.04, -0.02, 0.03]),
    )

    refined, iterations, rmse = refine_icp(
        source_points, target_points, initial,
        max_correspondence_distance=0.2,
        max_iterations=20,
    )

    assert iterations > 0
    assert rmse is not None and rmse < 1e-8
    assert refined.scale == initial.scale
    assert np.allclose(refined.translation, true_translation, atol=1e-8)
    assert np.allclose(refined.rotation, rotation, atol=1e-8)
