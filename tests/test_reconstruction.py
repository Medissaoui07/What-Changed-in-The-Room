"""Synthetic tests for imported point clouds and the replaceable backend contract."""

from pathlib import Path

import numpy as np
import pytest

from what_changed_room.reconstruction import (
    PointCloud,
    PointCloudFormatError,
    ReconstructionBackend,
    ReconstructionResult,
    import_point_cloud,
    load_point_cloud,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED_POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
EXPECTED_COLORS = np.eye(3)


@pytest.mark.parametrize("name", ["tiny.ply", "tiny.pcd"])
def test_imports_ascii_fixture(name: str) -> None:
    cloud = load_point_cloud(FIXTURES / name)
    np.testing.assert_allclose(cloud.points, EXPECTED_POINTS)
    np.testing.assert_allclose(cloud.colors, EXPECTED_COLORS)


def test_import_result_preserves_unknown_scale_and_source_frame() -> None:
    result = import_point_cloud(FIXTURES / "tiny.ply")
    assert isinstance(result, ReconstructionResult)
    assert result.coordinate_frame == "source:tiny.ply"
    assert result.scale_status == "unknown"
    assert result.metadata["format"] == "ply"
    assert len(result.point_cloud.points) == 3


def test_reads_binary_ply(tmp_path: Path) -> None:
    path = tmp_path / "binary.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
        + np.array([(1, 2, 3), (4, 5, 6)], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).tobytes()
    )
    np.testing.assert_allclose(load_point_cloud(path).points, [[1, 2, 3], [4, 5, 6]])


def test_reads_binary_pcd(tmp_path: Path) -> None:
    path = tmp_path / "binary.pcd"
    header = ("VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
              "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n").encode()
    path.write_bytes(header + np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4").tobytes())
    np.testing.assert_allclose(load_point_cloud(path).points, [[1, 2, 3], [4, 5, 6]])


def test_invalid_geometry_and_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        PointCloud(np.zeros((2, 2)))
    with pytest.raises(PointCloudFormatError, match="expected .ply or .pcd"):
        load_point_cloud(tmp_path / "scene.xyz")


def test_backend_contract_is_abstract() -> None:
    with pytest.raises(TypeError):
        ReconstructionBackend()
