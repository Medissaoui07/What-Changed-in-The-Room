"""Smoke tests for the initial package scaffold."""

import what_changed_room


def test_package_exposes_version() -> None:
    """The package can be imported and reports its installed version."""
    assert what_changed_room.__version__ == "0.1.0"
