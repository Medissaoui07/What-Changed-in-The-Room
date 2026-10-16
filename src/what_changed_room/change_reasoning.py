"""Conservative cross-capture association and candidate change reasoning.

Sparse keypoints inside an object box may belong to background. Consequently,
they are reported as shared-map context and never treated as object geometry.
Stronger movement/addition/removal candidates require explicit review evidence.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


_VISIBILITY = {"visible", "fully_visible", "occluded", "partially_visible", "unobserved", "unknown"}


def _read_json(path: str | Path, what: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {what} JSON {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{what} JSON must be an object with schema_version: 1")
    return value


def _track_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracks = report.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("observation report must contain a tracks array")
    result: dict[str, dict[str, Any]] = {}
    for track in tracks:
        if not isinstance(track, dict) or not isinstance(track.get("track_id"), str):
            raise ValueError("each track needs a track_id")
        track_id = track["track_id"]
        if track_id in result:
            raise ValueError(f"duplicate track_id in observation report: {track_id}")
        if track.get("capture") not in {"before", "after"}:
            raise ValueError(f"track {track_id} has invalid capture")
        if not isinstance(track.get("observations"), list):
            raise ValueError(f"track {track_id} needs an observations array")
        result[track_id] = track
    return result


def _point_ids(track: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for observation in track["observations"]:
        for point in observation.get("linked_3d_points", []):
            if isinstance(point, dict) and isinstance(point.get("point3d_id"), int):
                ids.add(point["point3d_id"])
    return ids


def _validate_anchor(value: object, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be [x, y, z] in before_colmap_map coordinates")
    try:
        anchor = np.asarray([float(component) for component in value], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric coordinates") from exc
    if not np.isfinite(anchor).all():
        raise ValueError(f"{field_name} must contain finite coordinates")
    return anchor


def _validate_review(review: dict[str, Any] | None, tracks: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if review is None:
        return [], []
    associations = review.get("associations", [])
    visibility = review.get("unmatched_visibility", [])
    if not isinstance(associations, list) or not isinstance(visibility, list):
        raise ValueError("review JSON associations and unmatched_visibility must be arrays")
    seen_pairs: set[tuple[str, str]] = set()
    for item in associations:
        if not isinstance(item, dict):
            raise ValueError("each review association must be an object")
        before_id, after_id = item.get("before_track_id"), item.get("after_track_id")
        if before_id not in tracks or after_id not in tracks:
            raise ValueError(f"review association references an unknown track: {before_id}, {after_id}")
        if tracks[before_id]["capture"] != "before" or tracks[after_id]["capture"] != "after":
            raise ValueError("association before/after track IDs must match their capture")
        pair = (before_id, after_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate review association: {pair}")
        seen_pairs.add(pair)
        if item.get("same_object", "unknown") not in {True, False, "unknown"}:
            raise ValueError("same_object must be true, false, or 'unknown'")
        for key in ("before_visibility", "after_visibility"):
            value = item.get(key, "unknown")
            if value not in _VISIBILITY:
                raise ValueError(f"{key} must be one of {sorted(_VISIBILITY)}")
        before_anchor = _validate_anchor(item.get("before_object_anchor_xyz"), "before_object_anchor_xyz")
        after_anchor = _validate_anchor(item.get("after_object_anchor_xyz"), "after_object_anchor_xyz")
        if (before_anchor is not None or after_anchor is not None) and item.get("anchor_source") != "manual_object_anchor":
            raise ValueError("reviewed object anchors require anchor_source: 'manual_object_anchor'")
    seen_unmatched: set[str] = set()
    for item in visibility:
        if not isinstance(item, dict) or item.get("track_id") not in tracks:
            raise ValueError("each unmatched_visibility entry must reference a known track_id")
        track_id = item["track_id"]
        if track_id in seen_unmatched:
            raise ValueError(f"duplicate unmatched visibility review for {track_id}")
        seen_unmatched.add(track_id)
        if item.get("opposite_region_visibility", "unknown") not in _VISIBILITY:
            raise ValueError("opposite_region_visibility must be a recognized visibility state")
        opposite = item.get("opposite_capture")
        expected = "after" if tracks[track_id]["capture"] == "before" else "before"
        if opposite != expected:
            raise ValueError(f"{track_id} opposite_capture must be {expected!r}")
    return associations, visibility


def compare_observation_reports(
    observations_path: str | Path,
    *,
    review_path: str | Path | None = None,
    movement_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare Phase 4 tracks, proposing links and retaining explicit unknowns.

    Matching one unique before and after track by class only proposes an
    association. Manual review may confirm identity and supply visibility or
    object-only map anchors. That prevents background sparse points from being
    mistaken for object surfaces or movement evidence.
    """
    if movement_threshold is not None and (not math.isfinite(movement_threshold) or movement_threshold <= 0):
        raise ValueError("movement_threshold must be finite and greater than zero")
    observations = _read_json(observations_path, "observation report")
    if observations.get("coordinate_frame") != "before_colmap_map":
        raise ValueError("observation report is not in the before COLMAP coordinate frame")
    tracks = _track_map(observations)
    review = None if review_path is None else _read_json(review_path, "review")
    reviewed_associations, unmatched_visibility = _validate_review(review, tracks)
    review_by_pair = {
        (item["before_track_id"], item["after_track_id"]): item
        for item in reviewed_associations
    }
    visibility_by_track = {item["track_id"]: item for item in unmatched_visibility}
    explicitly_used: set[str] = set()
    candidates: list[dict[str, Any]] = []

    by_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"before": [], "after": []})
    for track in tracks.values():
        label = str(track.get("label", "")).strip().casefold()
        by_class[label][track["capture"]].append(track)

    for label, captures in sorted(by_class.items()):
        before_tracks = sorted(captures["before"], key=lambda item: item["track_id"])
        after_tracks = sorted(captures["after"], key=lambda item: item["track_id"])
        if len(before_tracks) == 1 and len(after_tracks) == 1:
            before, after = before_tracks[0], after_tracks[0]
            pair = (before["track_id"], after["track_id"])
            explicitly_used.update(pair)
            review_item = review_by_pair.get(pair, {})
            same_object = review_item.get("same_object", "unknown")
            before_ids, after_ids = _point_ids(before), _point_ids(after)
            shared_ids = before_ids & after_ids
            # Sparse point overlap contextualizes the viewed region, but is not
            # object-specific identity or displacement evidence.
            candidate: dict[str, Any] = {
                "candidate_id": f"class:{label}",
                "label": before.get("label"),
                "candidate_type": "insufficient_observation",
                "observation_state": "observed_both",
                "association": {
                    "status": "review_confirmed_same_object" if same_object is True else "possible_same_class_match" if same_object == "unknown" else "review_says_different_objects",
                    "before_track_id": before["track_id"],
                    "after_track_id": after["track_id"],
                    "same_object": same_object,
                    "basis": ["same semantic label"] + (["manual identity review"] if same_object != "unknown" else []),
                },
                "geometry_evidence": {
                    "coordinate_frame": "before_colmap_map",
                    "before_linked_point_count": len(before_ids),
                    "after_linked_point_count": len(after_ids),
                    "shared_point_id_count": len(shared_ids),
                    "interpretation": "shared sparse points indicate overlapping mapped image regions; they may be background and are not object geometry",
                },
                "visibility": {
                    "before": review_item.get("before_visibility", "unknown"),
                    "after": review_item.get("after_visibility", "unknown"),
                },
                "evidence": {
                    "before_observations": before["observations"],
                    "after_observations": after["observations"],
                },
                "uncertainty_reasons": [
                    "same class does not establish object identity",
                    "Phase 4 boxes link sparse image keypoints, which may lie on background",
                    "no object-specific before/after position establishes a change",
                ],
            }
            before_anchor = _validate_anchor(review_item.get("before_object_anchor_xyz"), "before_object_anchor_xyz")
            after_anchor = _validate_anchor(review_item.get("after_object_anchor_xyz"), "after_object_anchor_xyz")
            if same_object is True and before_anchor is not None and after_anchor is not None:
                distance = float(np.linalg.norm(after_anchor - before_anchor))
                candidate["geometry_evidence"]["reviewed_object_anchor_displacement"] = distance
                candidate["geometry_evidence"]["anchor_source"] = "manual_object_anchor"
                if movement_threshold is None:
                    candidate["uncertainty_reasons"].append(
                        "object anchors are supplied, but no movement threshold was configured in sparse-map units"
                    )
                elif distance > movement_threshold:
                    candidate["candidate_type"] = "moved"
                    candidate["uncertainty_reasons"] = []
                else:
                    candidate["candidate_type"] = "observed_both"
                    candidate["uncertainty_reasons"] = []
            elif same_object is True:
                candidate["uncertainty_reasons"].append(
                    "identity was reviewed, but object-specific 3D anchors are unavailable"
                )
            elif same_object is False:
                candidate["candidate_type"] = "insufficient_observation"
                candidate["uncertainty_reasons"] = [
                    "review says these are different objects; addition/removal still requires opposite-region visibility evidence"
                ]
            candidates.append(candidate)
        else:
            # Ambiguous or unmatched same-class tracks do not become added or
            # removed candidates without explicit opposite-region visibility.
            for track in before_tracks + after_tracks:
                if track["track_id"] in explicitly_used:
                    continue
                visibility_item = visibility_by_track.get(track["track_id"], {})
                opposite_state = visibility_item.get("opposite_region_visibility", "unknown")
                can_classify = opposite_state == "fully_visible"
                is_before = track["capture"] == "before"
                candidates.append({
                    "candidate_id": f"track:{track['track_id']}",
                    "label": track.get("label"),
                    "candidate_type": (
                        ("candidate_removed" if is_before else "candidate_added")
                        if can_classify else "insufficient_observation"
                    ),
                    "observation_state": "observed_before" if track["capture"] == "before" else "observed_after",
                    "association": {"status": "ambiguous_or_unmatched", "track_id": track["track_id"]},
                    "visibility": {
                        "observed_capture": track["capture"],
                        "opposite_capture": "after" if is_before else "before",
                        "opposite_region": opposite_state,
                        "review_source": "manual" if visibility_item else None,
                    },
                    "evidence": {"observations": track["observations"]},
                    "uncertainty_reasons": [] if can_classify else [
                        "no reliable one-to-one cross-capture association",
                        "opposite region is not confirmed fully visible; unmatched detection does not prove addition or removal",
                    ],
                })

    return {
        "schema_version": 1,
        "coordinate_frame": "before_colmap_map",
        "source_observation_report": str(Path(observations_path).resolve()),
        "movement_threshold_map_units": movement_threshold,
        "policy": {
            "sparse_point_absence": "never proves removal",
            "sparse_points_in_box": "may be background; not object geometry",
            "same_class_association": "proposal only unless manually confirmed",
            "added_removed": "requires manual confirmation opposite region was fully visible",
            "movement": "requires same-object review, two object-specific manual anchors, and explicit threshold",
            "metric_scale": "COLMAP map units are arbitrary unless externally calibrated",
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def write_change_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write candidate reasoning as an atomic JSON report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
