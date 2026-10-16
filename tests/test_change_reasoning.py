"""Synthetic evidence-gating cases for Phase 5 change reasoning."""

import json

from what_changed_room.change_reasoning import compare_observation_reports


def _observation(point_ids: list[int]) -> dict[str, object]:
    return {
        "frame": "frame.jpg",
        "timestamp_seconds": 1.0,
        "label": "chair",
        "bbox_xyxy": [10, 20, 50, 80],
        "geometry_evidence": "sparse_keypoints_in_region" if point_ids else "image_space_only",
        "linked_3d_points": [
            {"point3d_id": point_id, "xyz": [float(point_id), 0.0, 0.0]}
            for point_id in point_ids
        ],
    }


def _report(*, before: bool = True, after: bool = True) -> dict[str, object]:
    tracks = []
    if before:
        tracks.append({
            "track_id": "before:chair",
            "capture": "before",
            "label": "chair",
            "observations": [_observation([1, 2, 3])],
        })
    if after:
        tracks.append({
            "track_id": "after:chair",
            "capture": "after",
            "label": "chair",
            "observations": [_observation([1, 2, 4])],
        })
    return {"schema_version": 1, "coordinate_frame": "before_colmap_map", "tracks": tracks}


def _write_report(tmp_path, value: dict[str, object]) -> str:
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_same_class_and_shared_points_do_not_claim_unchanged_or_moved(tmp_path):
    report = compare_observation_reports(_write_report(tmp_path, _report()))
    candidate = report["candidates"][0]
    assert candidate["candidate_type"] == "insufficient_observation"
    assert candidate["observation_state"] == "observed_both"
    assert candidate["geometry_evidence"]["shared_point_id_count"] == 2


def test_confirmed_identity_and_distinct_object_anchors_can_be_moved(tmp_path):
    review = {
        "schema_version": 1,
        "associations": [{
            "before_track_id": "before:chair",
            "after_track_id": "after:chair",
            "same_object": True,
            "before_object_anchor_xyz": [0.0, 0.0, 0.0],
            "after_object_anchor_xyz": [1.0, 0.0, 0.0],
            "anchor_source": "manual_object_anchor",
        }],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    report = compare_observation_reports(
        _write_report(tmp_path, _report()), review_path=review_path, movement_threshold=0.5
    )
    assert report["candidates"][0]["candidate_type"] == "moved"


def test_confirmed_nearby_anchors_stay_observed_both(tmp_path):
    review = {
        "schema_version": 1,
        "associations": [{
            "before_track_id": "before:chair",
            "after_track_id": "after:chair",
            "same_object": True,
            "before_object_anchor_xyz": [0.0, 0.0, 0.0],
            "after_object_anchor_xyz": [0.1, 0.0, 0.0],
            "anchor_source": "manual_object_anchor",
        }],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    report = compare_observation_reports(
        _write_report(tmp_path, _report()), review_path=review_path, movement_threshold=0.5
    )
    assert report["candidates"][0]["candidate_type"] == "observed_both"


def test_unmatched_track_is_not_removed_when_visibility_is_partial(tmp_path):
    review = {
        "schema_version": 1,
        "unmatched_visibility": [{
            "track_id": "before:chair",
            "opposite_capture": "after",
            "opposite_region_visibility": "partially_visible",
        }],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    report = compare_observation_reports(
        _write_report(tmp_path, _report(before=True, after=False)), review_path=review_path
    )
    assert report["candidates"][0]["candidate_type"] == "insufficient_observation"


def test_fully_visible_unmatched_track_is_only_a_removal_candidate(tmp_path):
    review = {
        "schema_version": 1,
        "unmatched_visibility": [{
            "track_id": "before:chair",
            "opposite_capture": "after",
            "opposite_region_visibility": "fully_visible",
        }],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    report = compare_observation_reports(
        _write_report(tmp_path, _report(before=True, after=False)), review_path=review_path
    )
    assert report["candidates"][0]["candidate_type"] == "candidate_removed"


def test_fully_visible_unmatched_after_track_is_only_an_addition_candidate(tmp_path):
    review = {
        "schema_version": 1,
        "unmatched_visibility": [{
            "track_id": "after:chair",
            "opposite_capture": "before",
            "opposite_region_visibility": "fully_visible",
        }],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    report = compare_observation_reports(
        _write_report(tmp_path, _report(before=False, after=True)), review_path=review_path
    )
    assert report["candidates"][0]["candidate_type"] == "candidate_added"


def test_viewpoint_only_sparse_overlap_cannot_prove_object_change(tmp_path):
    value = _report()
    value["tracks"][1]["observations"][0]["linked_3d_points"] = [
        {"point3d_id": point_id, "xyz": [float(point_id), 0.0, 0.0]}
        for point_id in [1, 2, 3]
    ]
    report = compare_observation_reports(_write_report(tmp_path, value))
    candidate = report["candidates"][0]
    assert candidate["geometry_evidence"]["shared_point_id_count"] == 3
    assert candidate["candidate_type"] == "insufficient_observation"
