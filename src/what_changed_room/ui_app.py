"""Local upload, reconstruction, and evidence-review UI for the room demo.

Run with ``streamlit run src/what_changed_room/ui_app.py`` from the repository
root. All uploads and outputs stay on this machine.
"""

from __future__ import annotations

import json
import os
import inspect
import importlib
import subprocess
import time
from pathlib import Path

import numpy as np
import streamlit as st

from what_changed_room.observations import UltralyticsSegDetector, annotations_from_detector
from what_changed_room.object_geometry import backproject_object_masks
from what_changed_room.openmvs_pipeline import reconstruct_openmvs_cpu
from what_changed_room.preprocessing import preprocess_pair
from what_changed_room.shared_map import _select_reference_model, _database_image_names, localize_after_images
from what_changed_room.visibility_evidence import build_visibility_evidence
from what_changed_room.object_matching import _masked_crop
from what_changed_room import textured_viewer

# Streamlit reruns this module in-process; reload this helper so a running UI
# does not keep emitting HTML from the previously imported WebGL renderer.
importlib.reload(textured_viewer)
write_textured_comparison = textured_viewer.write_textured_comparison
from what_changed_room.depth import (
    DepthAnythingV2SmallEstimator, anchors_for_frame, calibrate_relative_depth,
    calibration_as_dict, read_colmap_depth_anchors,
)
from what_changed_room.metric_scale import calibrate_scale_from_known_object


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / "output" / "ui_runs"
EMPTY_DEPTH = {"schema_version": 1, "frames": []}
YOLO_MODEL_ID = "yolo11n-seg.pt"
DINOV2_MODEL_ID = "facebook/dinov2-small"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
REFERENCE_TV_WIDTH_M = 0.90
REFERENCE_TV_HEIGHT_M = 0.55
OPENMVS_TOOLS = (
    "InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh",
    "RefineMesh", "TextureMesh", "Viewer",
)


def _default_openmvs_bin() -> str:
    candidates = (REPO / ".tools" / "openmvs-built" / "bin" / "OpenMVS", Path("/usr/local/bin"))
    for directory in candidates:
        if directory.is_dir() and all(
            any(path.is_file() and path.stat().st_mode & 0o111 for path in directory.rglob(name))
            for name in OPENMVS_TOOLS
        ):
            return str(directory)
    return "/usr/local/bin"


def _calibrate_run_scale(run: dict[str, object], device: str) -> dict[str, object]:
    """Calibrate this capture's scale from repeated known-size TV views."""
    annotations = json.loads(Path(run["annotations"]).read_text(encoding="utf-8"))
    tv = [item for item in annotations["observations"]
          if item.get("capture") == "before" and str(item.get("label", "")).casefold() in {"tv", "television"}]
    # Select one high-confidence TV mask per frame, favoring clear detections.
    by_frame: dict[str, dict[str, object]] = {}
    for item in tv:
        frame = str(item.get("frame"))
        if frame not in by_frame or float(item.get("score") or 0) > float(by_frame[frame].get("score") or 0):
            by_frame[frame] = item
    selected = sorted(by_frame.values(), key=lambda item: -float(item.get("score") or 0))[:5]
    if len(selected) < 2:
        return {"schema_version": 1, "status": "uncalibrated", "scale_m_per_map_unit": None,
                "reason": f"Need the known-size TV detected in at least two before frames; found {len(selected)}."}
    try:
        estimator = DepthAnythingV2SmallEstimator(device=device, model_path=DEPTH_MODEL_ID)
        image_anchors = read_colmap_depth_anchors(run["shared_model"])
        records = []
        output_dir = Path(run["job"]) / "tv_scale_depth"
        for item in selected:
            frame = str(item["frame"])
            relative = estimator.predict(Path(run["before_images"]) / frame)
            registered_name, anchors = anchors_for_frame(image_anchors, frame, "before")
            calibration = calibrate_relative_depth(relative, anchors)
            calibrated_path = None
            if calibration.status == "calibrated":
                camera_z = relative * float(calibration.scale) + float(calibration.offset)
                camera_z[camera_z <= 0] = np.nan
                calibrated_path_obj = output_dir / "camera_z" / f"{Path(frame).stem}.npy"
                calibrated_path_obj.parent.mkdir(parents=True, exist_ok=True)
                np.save(calibrated_path_obj, camera_z)
                calibrated_path = str(calibrated_path_obj.resolve())
            records.append({"capture": "before", "frame": frame, "registered_image_name": registered_name,
                            "calibration": calibration_as_dict(calibration),
                            "calibrated_camera_z_npy": calibrated_path})
        depth_report = {"schema_version": 1, "coordinate_frame": "camera_z_in_before_colmap_map_units", "frames": records}
        depth_path = output_dir / "depth_calibration.json"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.write_text(json.dumps(depth_report, indent=2) + "\n", encoding="utf-8")
        geometry_dir = Path(run["job"]) / "tv_scale_geometry"
        tv_geometry = backproject_object_masks(
            run["annotations"], depth_path, run["shared_model"], run["before_images"],
            run["after_images"], geometry_dir, triangulate_labels=(),
        )
        report = geometry_dir / "object_geometry.json"
        report_data = json.loads(report.read_text(encoding="utf-8"))
        # TV is the only mask required for this fit; retain the main run's geometry.
        for observation in report_data["observations"]:
            if str(observation.get("label", "")).casefold() not in {"tv", "television"}:
                observation["geometry_status"] = "image_space_only"
        report_data["observations"] = [x for x in report_data["observations"]
                                       if str(x.get("label", "")).casefold() in {"tv", "television"}]
        report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
        scale = calibrate_scale_from_known_object(
            report, label="tv", width_m=REFERENCE_TV_WIDTH_M, height_m=REFERENCE_TV_HEIGHT_M
        )
        scale["known_dimensions_source"] = "user-specified physical TV dimensions; recalibrated for this reconstruction"
        scale_path = Path(run["job"]) / "metric_scale_calibration.json"
        scale_path.write_text(json.dumps(scale, indent=2) + "\n", encoding="utf-8")
        return scale
    except Exception as exc:
        return {"schema_version": 1, "status": "uncalibrated", "scale_m_per_map_unit": None,
                "reason": f"Per-run TV scale calibration failed: {exc}"}


def _automatic_matches(run: dict[str, object], model_path: str, device: str) -> tuple[list[dict[str, object]], str]:
    """Compare per-capture object tracks using multi-view appearance and anchors.

    Detector boxes are frame observations, not object identities. We select one
    best observation per frame, ignore clipped low-confidence duplicates, and
    keep genuinely competing detections ambiguous.
    """
    report = json.loads(Path(run["annotations"]).read_text(encoding="utf-8"))
    observations = report.get("observations", [])
    centers = run["centers"]
    before_centers, after_centers = centers["before"], centers["after"]  # type: ignore[index]
    labels = sorted(
        set(before_centers) | set(after_centers)
        | {str(item.get("label", "")).casefold() for item in observations if item.get("label")}
    )
    if not labels:
        return [], "No class has a provisional 3D center in either capture."
    def without_identity_model(reason: str) -> tuple[list[dict[str, object]], str]:
        output = []
        for label in labels:
            b, a = before_centers.get(label), after_centers.get(label)
            before_seen = any(o.get("capture") == "before" and str(o.get("label", "")).casefold() == label for o in observations)
            after_seen = any(o.get("capture") == "after" and str(o.get("label", "")).casefold() == label for o in observations)
            if b and not after_seen:
                output.append({"label": label, "status": "before_only", "before": b,
                               "reason": "Possible removal candidate only; after visibility must be established."})
            elif a and not before_seen:
                output.append({"label": label, "status": "added", "after": a,
                               "reason": "Possible addition candidate only; verify before coverage."})
            else:
                output.append({"label": label, "status": "uncertain", "reason": reason})
        return output, reason
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        return without_identity_model("DINOv2 is unavailable; no identity is inferred from labels alone.")

    source = str(Path(model_path).expanduser()) if Path(model_path).expanduser().exists() else model_path
    try:
        processor = AutoImageProcessor.from_pretrained(source, local_files_only=Path(source).exists())
        model = AutoModel.from_pretrained(source, local_files_only=Path(source).exists())
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if selected_device.isdigit():
            selected_device = f"cuda:{selected_device}"
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            selected_device = "cpu"
        model.to(selected_device).eval()
    except Exception as exc:
        return without_identity_model(f"DINOv2 could not load ({exc}); identity remains uncertain.")

    image_dirs = {capture: Path(run[f"{capture}_images"]) for capture in ("before", "after")}
    track_frames = {
        (capture, label): set(map(str, center.get("frames", [])))
        for capture, capture_centers in centers.items()
        for label, center in capture_centers.items()
    }
    # Group detector observations into capture-local tracks. A track is grounded
    # by the triangulated center and spans distinct registered frames; an
    # individual detector box is never treated as a persistent object ID.
    by_track_frame: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for item in observations:
        capture, frame, label = str(item.get("capture")), str(item.get("frame")), str(item.get("label", "")).casefold()
        if capture in image_dirs and label and frame in track_frames.get((capture, label), set()):
            by_track_frame.setdefault((capture, label, frame), []).append(item)

    image_sizes: dict[Path, tuple[int, int] | None] = {}

    def observation_rank(item: dict[str, object], image_path: Path) -> tuple[float, float]:
        """Prefer a confident, fully visible mask over a clipped edge fragment."""
        score = float(item.get("score") or 0.0)
        box = item.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            return (0.0, score)
        try:
            import cv2
            if image_path not in image_sizes:
                image = cv2.imread(str(image_path))
                image_sizes[image_path] = None if image is None else (image.shape[1], image.shape[0])
            dimensions = image_sizes[image_path]
            if dimensions is None:
                return (score, 0.0)
            width, height = dimensions
            x1, y1, x2, y2 = (float(value) for value in box)
            margin = max(2.0, min(width, height) * 0.015)
            clipped = x1 <= margin or y1 <= margin or x2 >= width - 1 - margin or y2 >= height - 1 - margin
            return (0.0 if clipped else 1.0, score)
        except (ImportError, TypeError, ValueError):
            return (0.0, score)

    selected: dict[tuple[str, str], list[dict[str, object]]] = {}
    ambiguous_tracks: set[tuple[str, str]] = set()
    for (capture, label, frame), items in by_track_frame.items():
        image_path = image_dirs[capture] / frame
        ranked = sorted(items, key=lambda item: observation_rank(item, image_path), reverse=True)
        # A pair of high-confidence, non-clipped detections is evidence of
        # multiple instances; low-confidence border fragments are not.
        strong_interior = [item for item in ranked
                           if observation_rank(item, image_path)[0] > 0
                           and float(item.get("score") or 0.0) >= 0.45]
        if len(strong_interior) > 1:
            ambiguous_tracks.add((capture, label))
        selected.setdefault((capture, label), []).append(ranked[0])

    # Embed a balanced set of selected views from each track. The triangulated
    # frame list already excludes outlier observations; sampling it across time
    # avoids the old bias toward the first six, mostly clipped frames.
    track_crops: dict[tuple[str, str], list[tuple[dict[str, object], np.ndarray]]] = {}
    for track, items in selected.items():
        available = []
        for item in items:
            frame = str(item["frame"])
            crop = _masked_crop(item, image_dirs[track[0]] / frame)
            if crop is not None:
                available.append((item, crop))
        interior = [pair for pair in available
                    if observation_rank(pair[0], image_dirs[track[0]])[0] > 0]
        if len(interior) >= 2:
            # Ignore heavily truncated entrance/exit views for appearance
            # identity when the same triangulated track has enough full views.
            available = interior
        available.sort(key=lambda pair: str(pair[0]["frame"]))
        if len(available) > 12:
            indices = np.linspace(0, len(available) - 1, 12, dtype=int)
            available = [available[index] for index in sorted(set(indices))]
        track_crops[track] = available

    vectors: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}
    for track, entries in track_crops.items():
        crops = [crop for _, crop in entries]
        for start in range(0, len(crops), 8):
            inputs = processor(images=crops[start:start + 8], return_tensors="pt")
            inputs = {key: value.to(selected_device) for key, value in inputs.items()}
            with torch.inference_mode():
                batch = model(**inputs).last_hidden_state[:, 0].float()
                batch = torch.nn.functional.normalize(batch, dim=-1).cpu().numpy()
            vectors.setdefault(track, []).extend(
                (str(entries[start + index][0]["frame"]), vector) for index, vector in enumerate(batch)
            )

    result = []
    for label in labels:
        b, a = before_centers.get(label), after_centers.get(label)
        if not b or not a:
            before_class = any(o.get("capture") == "before" and str(o.get("label", "")).casefold() == label for o in observations)
            after_class = any(o.get("capture") == "after" and str(o.get("label", "")).casefold() == label for o in observations)
            if b and not after_class:
                result.append({"label": label, "status": "before_only", "before": b,
                               "reason": "No after detection or 3D track for this class. This is only a possible-removal candidate; visibility must be checked."})
            elif a and not before_class:
                result.append({"label": label, "status": "added", "after": a,
                               "reason": "An after-only multiview track was found. Confirm the same region was covered in the before capture before calling it an addition."})
            else:
                result.append({"label": label, "status": "uncertain", "reason": "A registered multiview center is missing in one capture; this cannot prove removal."})
            continue
        before_track, after_track = ("before", label), ("after", label)
        bv, av = vectors.get(before_track, []), vectors.get(after_track, [])
        if before_track in ambiguous_tracks or after_track in ambiguous_tracks:
            result.append({"label": label, "status": "uncertain", "reason": "Multiple confident, fully visible same-class objects occur in one capture frame; instance identity is ambiguous."})
            continue
        b_view_count = int((b.get("triangulation") or {}).get("view_count", 0))
        a_view_count = int((a.get("triangulation") or {}).get("view_count", 0))
        if b_view_count < 4 or a_view_count < 4 or len(bv) < 2 or len(av) < 2:
            result.append({"label": label, "status": "uncertain", "reason": "Need at least two usable views per capture for automatic identity matching."})
            continue
        # Compare all view pairs; the median is less sensitive to one poor crop
        # or a large viewpoint change than a mean of the first few detections.
        similarity_matrix = np.asarray([[float(np.dot(bvec, avec)) for _, avec in av] for _, bvec in bv])
        similarity = float(np.median(similarity_matrix))
        similarity_p25 = float(np.quantile(similarity_matrix, 0.25))
        representative_pair = np.unravel_index(
            int(np.argmin(np.abs(similarity_matrix - similarity))), similarity_matrix.shape
        )
        evidence_frames = {
            "before": bv[int(representative_pair[0])][0],
            "after": av[int(representative_pair[1])][0],
            "pair_cosine_similarity": float(similarity_matrix[representative_pair]),
        }
        displacement = float(np.linalg.norm(np.asarray(b["centroid_xyz_colmap"]) - np.asarray(a["centroid_xyz_colmap"])))
        # DINOv2 crop similarity drops under severe viewpoint/occlusion changes.
        # Use the cross-view median and lower quartile together; anchor tracks
        # must still be supported by >=4 registered views in both captures.
        identity_pass = similarity >= 0.60 and similarity_p25 >= 0.50
        status = "moved" if identity_pass and displacement > 0.08 else ("same_location" if identity_pass else "uncertain")
        result.append({
            "label": label, "status": status, "dinov2_cosine_similarity": similarity,
            "dinov2_similarity_p25": similarity_p25,
            "identity_threshold": 0.60, "identity_p25_threshold": 0.50,
            "displacement_colmap_units": displacement,
            "track_evidence": {"before_view_count": len(bv), "after_view_count": len(av),
                               "before_3d_view_count": b_view_count, "after_3d_view_count": a_view_count,
                               "before_feature_frames": [frame for frame, _ in bv],
                               "after_feature_frames": [frame for frame, _ in av],
                               "paired_rgb_frames": evidence_frames},
            "before": b, "after": a,
            "reason": (
                "Likely same object: each capture has one supported class track, multi-view anchors, "
                "and cross-view DINOv2 appearance above the demo cutoff. This is an automatic association, not proof of physical identity."
                if status != "uncertain" else
                "Track appearance or measured anchor shift was insufficient for an automatic conclusion."
            ),
        })
    del model
    if selected_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, "Each capture builds a track from its per-frame detections and provisional multiview anchor. DINOv2 compares several selected views; confident competing instances and weak evidence remain uncertain."


def _write_upload(uploaded: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded.getvalue())  # type: ignore[attr-defined]


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n" + result.stdout + "\n")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout[-5000:]}")


def _prepare_capture(
    before_video: Path, after_video: Path, job: Path, *, interval: float,
    colmap: str, model_path: str, device: str, openmvs_bin: str,
) -> dict[str, object]:
    frames_root = job / "frames"
    preprocess_pair(before_video, after_video, frames_root, sample_interval_seconds=interval)
    before_images, after_images = frames_root / "before", frames_root / "after"
    colmap_root = job / "colmap"
    colmap_root.mkdir(parents=True, exist_ok=True)
    database = colmap_root / "before.db"
    sparse_parent = colmap_root / "sparse"
    sparse_parent.mkdir(exist_ok=True)
    log_path = job / "logs" / "pipeline.log"
    _run([colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(before_images),
          "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", "0"], REPO, log_path)
    _run([colmap, "exhaustive_matcher", "--database_path", str(database), "--SiftMatching.use_gpu", "0"], REPO, log_path)
    _run([colmap, "mapper", "--database_path", str(database), "--image_path", str(before_images),
          "--output_path", str(sparse_parent)], REPO, log_path)
    reference_model = _select_reference_model(sparse_parent, set(_database_image_names(database)))

    shared_dir = job / "shared_map"
    localization = localize_after_images(reference_model, database, after_images, shared_dir, colmap_executable=colmap)
    shared_model = localization.get("shared_model")
    annotations_path = job / "annotations.json"
    frames = {
        "before": sorted(before_images.glob("*.jpg")),
        "after": sorted(after_images.glob("*.jpg")),
    }
    detector = UltralyticsSegDetector(model_path, device=device, preview_dir=job / "previews")
    annotations = annotations_from_detector(detector, frames)
    annotations_path.write_text(json.dumps(annotations, indent=2) + "\n", encoding="utf-8")

    geometry_path: Path | None = None
    centers: dict[str, dict[str, dict[str, object]]] = {"before": {}, "after": {}}
    if shared_model:
        depth_path = job / "depth_not_run.json"
        depth_path.write_text(json.dumps(EMPTY_DEPTH), encoding="utf-8")
        geometry_dir = job / "geometry"
        geometry = backproject_object_masks(
            annotations_path, depth_path, shared_model, before_images, after_images, geometry_dir,
            triangulate_labels=tuple(sorted({str(item.get("label", "")).casefold() for item in annotations["observations"]})),
        )
        geometry_path = geometry_dir / "object_geometry.json"
        for item in geometry["observations"]:
            if item.get("geometry_status") != "triangulated_provisional":
                continue
            capture, label = str(item["capture"]), str(item["label"]).casefold()
            capture_data = centers[capture]
            if label not in capture_data:
                capture_data[label] = {
                    "centroid_xyz_colmap": item["centroid_xyz_colmap"],
                    "triangulation": item.get("triangulation", {}),
                    "frames": item.get("triangulation_track_frames", []),
                }
    return {
        "job": str(job), "frames_root": str(frames_root), "before_images": str(before_images),
        "after_images": str(after_images), "reference_model": str(reference_model),
        "database": str(database), "shared_model": shared_model,
        "localization": localization, "annotations": str(annotations_path),
        "geometry": None if geometry_path is None else str(geometry_path),
        "centers": centers, "openmvs_bin": str(Path(openmvs_bin).resolve()),
    }


def _candidate_report(run: dict[str, object]) -> dict[str, object]:
    matches = run.get("automatic_matches", [])
    report: dict[str, object] = {
        "schema_version": 1, "coordinate_frame": "shared_before_colmap_map",
        "metric_scale": bool(run.get("meters_per_map_unit")), "candidates": [],
        "automatic_matching": run.get("automatic_matching_note"),
    }
    candidates: list[dict[str, object]] = []
    for item in matches:
        label = str(item["label"])
        b, a = item.get("before"), item.get("after")
        if item.get("status") in {"moved", "same_location"} and b and a:
            candidates.append({
                "candidate_id": f"{label}:automatic_same_instance", "label": label,
                "candidate_type": "same_object_reviewed",
                "before": {"centroid_xyz_colmap": b["centroid_xyz_colmap"], "frames": b["frames"]},
                "after": {"centroid_xyz_colmap": a["centroid_xyz_colmap"], "frames": a["frames"]},
                "review": {"same_object": True, "geometry_accepted": True, "decision_source": "automatic_model_match"},
                "automatic_match": {
                    "method": "multi-view DINOv2 track appearance and triangulated anchors",
                    "cosine_similarity_median": item.get("dinov2_cosine_similarity"),
                    "cosine_similarity_p25": item.get("dinov2_similarity_p25"),
                    "threshold": item.get("identity_threshold"),
                    "track_evidence": item.get("track_evidence"),
                },
                "evidence_note": "Automatically matched from multiple masked views and provisional shared-map anchors; verify paired evidence before relying on it.",
            })
        elif item.get("status") == "before_only" and b:
            candidates.append({
                "candidate_id": f"{label}:before_only", "label": label,
                "candidate_type": "unmatched_before",
                "before": {"centroid_xyz_colmap": b["centroid_xyz_colmap"], "frames": b["frames"]},
                "review": {"same_object": None, "visibility": "unknown"},
                "evidence_note": "No after correspondence; absence is not proof of removal. Inspect the projected old footprint in several views.",
            })
        elif item.get("status") == "added" and a:
            candidates.append({
                "candidate_id": f"{label}:after_only", "label": label,
                "candidate_type": "unmatched_after",
                "after": {"centroid_xyz_colmap": a["centroid_xyz_colmap"], "frames": a["frames"]},
                "review": {"same_object": None, "visibility": "unknown"},
                "evidence_note": str(item.get("reason", "After-only track; verify before visibility.")),
            })
        else:
            candidates.append({
                "candidate_id": f"{label}:uncertain", "label": label,
                "candidate_type": "insufficient_observation",
                **({"before": {"centroid_xyz_colmap": b["centroid_xyz_colmap"], "frames": b["frames"]}} if b else {}),
                "review": {"visibility": "unknown"},
                "evidence_note": str(item.get("reason", "Insufficient evidence.")),
            })
    report["candidates"] = candidates
    return report


def _movement_rows(run: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for candidate in _candidate_report(run)["candidates"]:
        if candidate["candidate_type"] != "same_object_reviewed":
            continue
        before_xyz = candidate["before"]["centroid_xyz_colmap"]
        after_xyz = candidate["after"]["centroid_xyz_colmap"]
        map_distance = float(np.linalg.norm(np.asarray(before_xyz) - np.asarray(after_xyz)))
        rows.append({
            "label": candidate["label"], "before_xyz_colmap": before_xyz,
            "after_xyz_colmap": after_xyz,
            "estimated_displacement_m": map_distance * run["meters_per_map_unit"] if run.get("meters_per_map_unit") else None,
        })
    return rows


def _launch_viewer(executable: Path, scene: Path, texture_mesh: Path) -> None:
    env = os.environ.copy()
    library_dir = REPO / ".tools" / "openmvs-built" / "lib"
    env["LD_LIBRARY_PATH"] = str(library_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.Popen([str(executable), "--input-file", str(scene), "--geometry-file", str(texture_mesh)], env=env)


def main() -> None:
    st.set_page_config(page_title="Room Change Lab", layout="wide")
    st.title("Room Change Lab")
    st.caption("Upload two room recordings. The app builds paired RGB reconstructions and reports object changes with frame evidence.")
    upload_tab, results_tab = st.tabs(["1 · Upload & process", "2 · Results"])

    with upload_tab:
        st.markdown("Keep the room landmarks visible in both recordings. Move or remove a few objects while recording overlapping views.")
        before_upload = st.file_uploader("Before video", type=["mp4", "mov", "m4v", "avi"], key="before_upload")
        after_upload = st.file_uploader("After video", type=["mp4", "mov", "m4v", "avi"], key="after_upload")
        left, right = st.columns(2)
        with left:
            interval = st.slider("Sample interval (seconds)", 0.5, 3.0, 1.5, 0.5)
            colmap = st.text_input("COLMAP executable", "colmap")
            device = st.selectbox("AI device", ["auto", "0", "cpu"])
            openmvs_bin = st.text_input("OpenMVS tools folder", _default_openmvs_bin())
        with right:
            st.info("The default YOLO11n-seg, DINOv2 Small, and Depth Anything V2 Small models load automatically. Internet is needed for their first download. TV scale calibration assumes 0.90 × 0.55 m; if your TV differs, metric scale may be wrong. Insufficient evidence leaves distances in map units.")
            st.warning("DINOv2 automatically links only clear, unambiguous matches. Weak matches remain uncertain. An object detector missing an object never proves removal.")
        if st.button("Process both videos", type="primary", disabled=not (before_upload and after_upload)):
            job = OUTPUT_ROOT / time.strftime("capture_%Y%m%d_%H%M%S")
            job.mkdir(parents=True, exist_ok=False)
            before_video, after_video = job / "uploads" / "before.mp4", job / "uploads" / "after.mp4"
            _write_upload(before_upload, before_video)
            _write_upload(after_upload, after_video)
            status = st.status("Starting local reconstruction…", expanded=True)
            try:
                result = _prepare_capture(
                    before_video, after_video, job, interval=interval, colmap=colmap,
                    model_path=YOLO_MODEL_ID, device=device, openmvs_bin=openmvs_bin,
                )
                matches, match_note = _automatic_matches(result, DINOV2_MODEL_ID, device)
                result["automatic_matches"] = matches
                result["automatic_matching_note"] = match_note
                scale = {"status": "uncalibrated", "scale_m_per_map_unit": None,
                         "reason": "TV scale calibration has not been run."}
                if result.get("shared_model"):
                    scale = _calibrate_run_scale(result, device)
                result["metric_scale"] = scale
                result["meters_per_map_unit"] = scale.get("scale_m_per_map_unit") if scale.get("status") == "provisional" else None
                for key in list(st.session_state):
                    if key.startswith(("removal_result_", "movement_evidence_", "visibility_")):
                        st.session_state.pop(key, None)
                st.session_state["active_run"] = result
                st.session_state.pop("map_result", None)
                st.session_state.pop("map_attempted", None)
                st.session_state.pop("map_error", None)
                st.session_state["reviewed_report"] = None
                status.update(label="Videos processed. Automatic findings are ready.", state="complete")
                st.success(f"Run saved in {job}")
            except Exception as exc:
                status.update(label="Pipeline stopped", state="error")
                st.exception(exc)
                st.info(f"Completed stages and logs are preserved under {job}")

    with results_tab:
        run = st.session_state.get("active_run")
        if not run:
            st.info("Upload and process a before/after video pair in the first tab.")
            return
        localization = run["localization"]
        st.header("Results: what changed")
        counts = {k: localization.get(k) for k in ("before_registered", "after_registered", "points", "localization_success")}
        st.caption(f"Shared coordinate frame · registered cameras: before {counts['before_registered']}, after {counts['after_registered']} · sparse landmarks: {counts['points']}")
        scale = run.get("metric_scale", {})
        reference = scale.get("reference_object", {}).get("dimensions_m", {}) if isinstance(scale, dict) else {}
        if run.get("meters_per_map_unit"):
            st.success(f"Per-run TV calibration: 1 COLMAP map unit = {run['meters_per_map_unit']:.5f} m (provisional; TV {reference.get('width', 0):.2f} × {reference.get('height', 0):.2f} m).")
        else:
            st.warning(f"Distances remain in COLMAP map units. {scale.get('reason', 'Scale calibration is unavailable for this run.')}")
        st.caption(str(run.get("automatic_matching_note", "")))

        matches = run.get("automatic_matches", [])
        report_path = Path(run["job"]) / "automatic_candidates.json"
        if not report_path.exists():
            report_path.write_text(json.dumps(_candidate_report(run), indent=2) + "\n", encoding="utf-8")
        st.session_state["reviewed_report"] = str(report_path)
        if matches:
            st.subheader("Detected changes")
            for item in matches:
                label, finding = str(item["label"]), str(item["status"])
                title = {"moved": f"Moved · {label}", "same_location": f"Observed · {label}",
                         "before_only": f"Possible removal · {label}", "added": f"Possible addition · {label}",
                         "uncertain": f"Uncertain · {label}"}.get(finding, f"{finding} · {label}")
                with st.container():
                    st.subheader(title)
                    st.write(item.get("reason", ""))
                    if item.get("dinov2_cosine_similarity") is not None:
                        st.write(
                            f"Multi-view DINOv2 similarity: median {item['dinov2_cosine_similarity']:.3f} "
                            f"(cutoff {item.get('identity_threshold', 0.60):.2f}); "
                            f"25th percentile {item.get('dinov2_similarity_p25', 0.0):.3f} "
                            f"(cutoff {item.get('identity_p25_threshold', 0.50):.2f})."
                        )
                        evidence = item.get("track_evidence", {})
                        if evidence:
                            st.caption(
                                f"Compared {evidence.get('before_view_count', 0)} before and "
                                f"{evidence.get('after_view_count', 0)} after views; "
                                f"3D anchors used {evidence.get('before_3d_view_count', 0)} and "
                                f"{evidence.get('after_3d_view_count', 0)} views."
                            )
                    if item.get("status") in {"moved", "same_location"}:
                        distance = float(item.get("displacement_colmap_units", 0.0))
                        if run.get("meters_per_map_unit"):
                            st.write(f"Center displacement: approximately **{distance * run['meters_per_map_unit']:.2f} m**.")
                        else:
                            st.write(f"Center displacement: **{distance:.3f} COLMAP map units**; no meter claim is available.")
                    elif finding == "before_only":
                        st.write("This is a candidate only. Removal requires the old full footprint to be visible and empty from several after views.")
                    elif finding == "added":
                        st.write("This is a candidate only. The before capture must have clearly observed this region before calling it an addition.")
                    b_frames = (item.get("before") or {}).get("frames", [])
                    a_frames = (item.get("after") or {}).get("frames", [])
                    paired = (item.get("track_evidence") or {}).get("paired_rgb_frames", {})
                    if paired.get("before") and paired.get("after"):
                        evidence_pairs = [(str(paired["before"]), str(paired["after"]))]
                    else:
                        evidence_pairs = list(zip(map(str, b_frames[:3]), map(str, a_frames[:3])))
                    pairs = len(evidence_pairs)
                    if pairs:
                        st.markdown("**Paired RGB evidence** (mask overlay previews):")
                        for before_frame, after_frame in evidence_pairs:
                            columns = st.columns(2)
                            for col, capture, frame in ((columns[0], "before", before_frame), (columns[1], "after", after_frame)):
                                preview = Path(run["job"]) / "previews" / frame
                                if preview.is_file():
                                    col.image(str(preview), caption=f"{capture.title()} · {frame}")
                    else:
                        st.caption("No paired mask-preview frames are available for this candidate.")
                        if finding in {"before_only", "removed"} and b_frames:
                            first_before = Path(run["job"]) / "previews" / str(b_frames[0])
                            if first_before.is_file():
                                st.image(str(first_before), caption=f"Before evidence · {label} · {b_frames[0]}")
                    if finding in {"moved", "same_location"} and run.get("shared_model"):
                        gallery_dir = Path(run["job"]) / "visibility" / f"{label.replace(' ', '_')}_movement"
                        evidence_json = gallery_dir / "visibility_evidence.json"
                        if not evidence_json.is_file():
                            build_visibility_evidence(
                                report_path, run["annotations"], run["shared_model"],
                                run["after_images"], gallery_dir,
                                candidate_id=f"{label}:automatic_same_instance",
                            )
                        if evidence_json.is_file():
                            projected = json.loads(evidence_json.read_text(encoding="utf-8"))
                            rows = [row for row in projected["frames"]
                                    if row["old_center_inside_image"] or row["new_center_inside_image"]][:3]
                            if rows:
                                st.markdown("**Projected movement centers in registered after views**")
                                columns = st.columns(len(rows))
                                import cv2
                                for col, row in zip(columns, rows):
                                    image = cv2.imread(row["image"])
                                    if image is None:
                                        continue
                                    for point, color, caption in (
                                        (row["projected_old_center_xy"], (0, 30, 255), "old center"),
                                        (row["projected_new_center_xy"], (40, 190, 40), "new center"),
                                    ):
                                        if point is not None:
                                            xy = (round(point[0]), round(point[1]))
                                            cv2.circle(image, xy, 11, color, 3)
                                            cv2.putText(image, caption, (xy[0] + 12, xy[1]),
                                                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
                                    for box in row["current_same_class_boxes_xyxy"]:
                                        x1, y1, x2, y2 = map(round, box)
                                        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 190, 40), 2)
                                    col.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                              caption=f"{row['frame']} · red old / green new")

        st.subheader("Paired RGB room maps")
        if not run.get("shared_model") or int(localization.get("after_registered", 0)) < 3:
            st.error("Fewer than three after cameras registered. Improve viewpoint overlap before building the paired RGB maps.")
        elif not st.session_state.get("map_result") and not st.session_state.get("map_attempted"):
            st.session_state["map_attempted"] = True
            job = Path(run["job"])
            maps = job / "textured_maps"
            attempt = 1
            while maps.exists():
                maps = job / f"textured_maps_attempt_{attempt:02d}"
                attempt += 1
            try:
                with st.spinner("OpenMVS is estimating and texturing both room surfaces on CPU. This may take several minutes…"):
                    reconstruct_kwargs = {}
                    if "meters_per_map_unit" in inspect.signature(reconstruct_openmvs_cpu).parameters:
                        reconstruct_kwargs["meters_per_map_unit"] = run.get("meters_per_map_unit")
                    result = reconstruct_openmvs_cpu(
                        run["shared_model"], run["before_images"], run["after_images"],
                        report_path, run["openmvs_bin"], maps, **reconstruct_kwargs,
                    )
                # Keep runs working if Streamlit has an older imported
                # openmvs_pipeline module loaded from before a restart.
                result_report = Path(result["report"])
                if result_report.is_file():
                    openmvs_report = json.loads(result_report.read_text(encoding="utf-8"))
                    scale_factor = run.get("meters_per_map_unit")
                    openmvs_report["metric_scale"] = scale_factor is not None
                    openmvs_report["meters_per_map_unit"] = scale_factor
                    for movement in openmvs_report.get("object_movement", []):
                        movement["estimated_displacement_m"] = (
                            movement.get("displacement_map_units", 0.0) * scale_factor
                            if scale_factor is not None else None
                        )
                    result_report.write_text(json.dumps(openmvs_report, indent=2) + "\n", encoding="utf-8")
                movement_rows = _movement_rows(run)
                html_path = Path(result["output_dir"]) / "paired_rgb_maps.html"
                write_textured_comparison(
                    Path(result["output_dir"]) / "before" / "scene_textured.ply",
                    Path(result["output_dir"]) / "after" / "scene_textured.ply",
                    html_path, movements=movement_rows,
                    meters_per_map_unit=run.get("meters_per_map_unit"),
                    shared_model=run.get("shared_model"),
                )
                result["paired_rgb_maps_html"] = str(html_path)
                st.session_state["map_result"] = result
            except Exception as exc:
                st.session_state["map_error"] = str(exc)
        map_result = st.session_state.get("map_result")
        if not map_result and st.session_state.get("map_error"):
            st.error(f"Paired RGB map generation failed: {st.session_state['map_error']}")
        if map_result:
            viewer_html = Path(map_result.get("paired_rgb_maps_html", ""))
            if viewer_html.is_file():
                st.components.v1.html(viewer_html.read_text(encoding="utf-8"), height=800, scrolling=True)
            else:
                st.error("The paired RGB map viewer was not generated.")
            st.caption(f"Reconstruction outputs: {map_result['output_dir']}")

        st.subheader("Possible removals · visibility evidence")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        old_only = [c for c in report["candidates"] if c["candidate_type"] == "unmatched_before"]
        if not old_only:
            st.write("No before-only object tracks were proposed. This does not imply that removals are impossible; detector coverage is limited.")
        elif run.get("shared_model"):
            st.caption("Projected old centers are shown automatically in up to three registered after views per candidate. This is evidence for inspection only; the system does not automatically confirm removals.")
            import cv2
            for candidate in old_only:
                label = str(candidate["label"])
                st.markdown(f"#### Possible removal · {label}")
                gallery_dir = Path(run["job"]) / "visibility" / label.replace(" ", "_")
                evidence_path = gallery_dir / "visibility_evidence.json"
                if not evidence_path.is_file():
                    build_visibility_evidence(
                        report_path, run["annotations"], run["shared_model"],
                        run["after_images"], gallery_dir,
                        candidate_id=candidate["candidate_id"],
                    )
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                review_rows = [row for row in evidence["frames"] if row["old_center_inside_image"]]
                if not review_rows:
                    st.info("No registered after view projects the old center inside the image, so visibility cannot be assessed from these captures.")
                    continue
                selected_rows = review_rows[:3]
                columns = st.columns(len(selected_rows))
                for col, row in zip(columns, selected_rows):
                    image = cv2.imread(row["image"])
                    if image is None:
                        continue
                    px, py = row["projected_old_center_xy"]
                    cv2.circle(image, (round(px), round(py)), 11, (0, 30, 255), 3)
                    cv2.putText(image, "old center", (round(px) + 12, round(py)),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 30, 255), 2)
                    for box in row["current_same_class_boxes_xyxy"]:
                        x1, y1, x2, y2 = map(round, box)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (20, 190, 40), 2)
                    col.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                              caption=f"{row['frame']} · red old center / green detections")
                if len(review_rows) < 3:
                    st.warning(f"Only {len(review_rows)} registered views include the projected old center; evidence is insufficient to assess removal.")


if __name__ == "__main__":
    main()
