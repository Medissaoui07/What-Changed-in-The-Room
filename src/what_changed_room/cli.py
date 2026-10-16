"""Command-line entry point for video preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .colmap_sparse import load_colmap_sparse_model, match_colmap_sparse_models
from .change_reasoning import compare_observation_reports, write_change_report
from .depth import (
    DepthAnythingV2SmallEstimator,
    anchors_for_frame,
    calibration_as_dict,
    calibrate_relative_depth,
    read_colmap_depth_anchors,
    write_depth_preview,
    write_depth_report,
)
from .observations import (
    UltralyticsSegDetector,
    annotations_from_detector,
    build_observation_report,
    write_observation_report,
)
from .object_geometry import backproject_object_masks
from .object_matching import (
    DEFAULT_DINO_MODEL,
    compare_reviewed_object_matches,
    rank_object_matches,
)
from .openmvs_pipeline import reconstruct_openmvs_cpu
from .preprocessing import VideoValidationError, preprocess_pair
from .reconstruction import PointCloudFormatError, import_point_cloud
from .registration import register_point_clouds
from .room_reconstruction import reconstruct_sparse_room
from .shared_map import localize_after_images
from .visibility_evidence import build_visibility_evidence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(prog="wcr", description="Preprocess a before/after video pair")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preprocess = subparsers.add_parser("preprocess", help="extract frames and write a manifest")
    preprocess.add_argument("before", help="path to the before video")
    preprocess.add_argument("after", help="path to the after video")
    preprocess.add_argument("--output-dir", required=True, help="directory for frames and manifest")
    preprocess.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="target time between sampled frames in seconds (default: 1.0)",
    )
    preprocess.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="optional maximum number of frames extracted per video",
    )
    point_cloud = subparsers.add_parser("import-point-cloud", help="inspect an existing PLY or PCD reconstruction")
    point_cloud.add_argument("path", help="path to a .ply or .pcd point cloud")
    registration = subparsers.add_parser(
        "register-colmap", help="align two COLMAP 3.x sparse models using tracked SIFT features"
    )
    registration.add_argument("before_model", help="native COLMAP model folder for the before scene")
    registration.add_argument("after_model", help="native COLMAP model folder for the after scene")
    registration.add_argument("--before-database", required=True, help="before COLMAP database.db")
    registration.add_argument("--after-database", required=True, help="after COLMAP database.db")
    registration.add_argument("--descriptor-ratio", type=float, default=0.8)
    registration.add_argument("--inlier-threshold", type=float, default=0.03,
                               help="RANSAC threshold in before-model coordinate units")
    registration.add_argument("--max-iterations", type=int, default=1000)
    registration.add_argument("--icp-iterations", type=int, default=20)
    registration.add_argument("--scale-known", action="store_true",
                               help="fix scale to 1 when both input models have known matching metric scale")
    localization = subparsers.add_parser(
        "localize-colmap",
        help="localize after frames into the before COLMAP map and write a visual preview",
    )
    localization.add_argument(
        "before_model",
        help="COLMAP binary model folder, or sparse/ parent (best submodel is selected automatically)",
    )
    localization.add_argument("--before-database", required=True, help="database used to create the reference map")
    localization.add_argument(
        "--before-images",
        help=argparse.SUPPRESS,
    )
    localization.add_argument("--after-images", required=True, help="directory containing after keyframes")
    localization.add_argument("--output-dir", required=True, help="where to save the shared model and HTML preview")
    localization.add_argument("--colmap", default="colmap", help="COLMAP executable (default: colmap)")
    sparse_room = subparsers.add_parser(
        "reconstruct-sparse-room",
        help="build a rotatable room-structure preview from sparse COLMAP points and reviewed object anchors",
    )
    sparse_room.add_argument("--shared-model", required=True, help="shared COLMAP binary model")
    sparse_room.add_argument("--object-report", help="optional reviewed object-match JSON")
    sparse_room.add_argument("--output-dir", required=True, help="where to write the HTML and evidence JSON")
    openmvs = subparsers.add_parser(
        "reconstruct-openmvs-cpu",
        help="densify a shared sparse map into separate CPU textured before/after models",
    )
    openmvs.add_argument("--shared-model", required=True, help="shared COLMAP binary model")
    openmvs.add_argument("--before-images", required=True, help="before RGB frame directory")
    openmvs.add_argument("--after-images", required=True, help="after RGB frame directory")
    openmvs.add_argument("--object-report", required=True, help="reviewed object match JSON")
    openmvs.add_argument("--openmvs-bin", required=True, help="OpenMVS binary directory")
    openmvs.add_argument("--output-dir", required=True, help="destination for models and logs")
    openmvs.add_argument("--max-threads", type=int, default=8)
    observations = subparsers.add_parser(
        "observations",
        help="track manual/model object observations and link supported regions to sparse 3D points",
    )
    observations.add_argument("annotations", help="JSON annotations/detections for selected RGB frames")
    observations.add_argument("--shared-model", required=True, help="COLMAP model containing both camera sets")
    observations.add_argument("--before-images", required=True, help="selected before RGB frame directory")
    observations.add_argument("--after-images", required=True, help="selected after RGB frame directory")
    observations.add_argument("--frame-manifest", help="preprocessing frame_manifest.json for timestamps")
    observations.add_argument("--output", required=True, help="destination observations JSON report")
    detect = subparsers.add_parser(
        "detect-objects",
        help="propose object masks on selected frames using an Ultralytics segmentation model",
    )
    detect.add_argument(
        "--frame",
        action="append",
        required=True,
        metavar="CAPTURE:FILENAME",
        help="selected frame, e.g. before:before_000019.jpg (repeat for each frame)",
    )
    detect.add_argument("--before-images", required=True, help="before RGB frame directory")
    detect.add_argument("--after-images", required=True, help="after RGB frame directory")
    detect.add_argument("--model", default="yolo11n-seg.pt", help="YOLO segmentation checkpoint")
    detect.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index (default: auto)")
    detect.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    detect.add_argument("--imgsz", type=int, default=640, help="inference image size")
    detect.add_argument("--preview-dir", help="optional directory for annotated mask previews")
    detect.add_argument("--output", required=True, help="destination annotation proposals JSON")
    estimate_depth = subparsers.add_parser(
        "estimate-depth",
        help="predict relative depth and calibrate selected registered frames to COLMAP camera Z",
    )
    estimate_depth.add_argument(
        "--frame",
        action="append",
        required=True,
        metavar="CAPTURE:FILENAME",
        help="selected frame, e.g. after:after_000008.jpg (repeat for each frame)",
    )
    estimate_depth.add_argument("--before-images", required=True, help="before RGB frame directory")
    estimate_depth.add_argument("--after-images", required=True, help="after RGB frame directory")
    estimate_depth.add_argument("--shared-model", required=True, help="registered COLMAP binary model")
    estimate_depth.add_argument("--device", default="auto", help="auto, cpu, or cuda (default: auto)")
    estimate_depth.add_argument(
        "--model-path",
        help="local Hugging Face model directory (or leave unset to load the official model ID online)",
    )
    estimate_depth.add_argument("--output-dir", required=True, help="directory for depth arrays and diagnostics")
    backproject = subparsers.add_parser(
        "backproject-objects",
        help="back-project polygon masks from frames with accepted depth calibration into the shared COLMAP map",
    )
    backproject.add_argument("annotations", help="YOLO or manual annotation JSON with polygon masks")
    backproject.add_argument("--depth-report", required=True, help="estimate-depth depth_calibration.json")
    backproject.add_argument("--shared-model", required=True, help="shared COLMAP binary model")
    backproject.add_argument("--before-images", required=True, help="before RGB frame directory")
    backproject.add_argument("--after-images", required=True, help="after RGB frame directory")
    backproject.add_argument("--output-dir", required=True, help="where to write object_geometry.json and objects.ply")
    backproject.add_argument("--max-points-per-mask", type=int, default=2500)
    backproject.add_argument("--minimum-points-per-mask", type=int, default=20)
    backproject.add_argument(
        "--triangulate-label",
        action="append",
        default=[],
        help="optionally estimate provisional 3D object-center anchors from repeated registered mask centers (repeatable)",
    )
    rank_matches = subparsers.add_parser(
        "rank-object-matches",
        help="rank cross-capture object-mask pairs by DINOv2 appearance similarity for human review",
    )
    rank_matches.add_argument("geometry", help="backproject-objects object_geometry.json")
    rank_matches.add_argument("--before-images", required=True, help="before RGB frame directory")
    rank_matches.add_argument("--after-images", required=True, help="after RGB frame directory")
    rank_matches.add_argument("--output-dir", required=True, help="directory for crops, JSON rankings, and HTML review page")
    rank_matches.add_argument("--model", default=DEFAULT_DINO_MODEL, help="DINOv2 model ID or local Transformers model directory")
    rank_matches.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index (default: auto)")
    rank_matches.add_argument("--top-k", type=int, default=5, help="after candidates to keep for each before mask")
    rank_matches.add_argument("--batch-size", type=int, default=8)
    compare_matches = subparsers.add_parser(
        "compare-reviewed-matches",
        help="create conservative movement candidates from explicit object identity and geometry review",
    )
    compare_matches.add_argument("geometry", help="object_geometry.json")
    compare_matches.add_argument("rankings", help="DINO dino_match_candidates.json")
    compare_matches.add_argument("review", help="review JSON exported from dino_match_review.html")
    compare_matches.add_argument("--movement-threshold", type=float, help="required map-unit displacement threshold to label a moved candidate")
    compare_matches.add_argument(
        "--track-distance-threshold", type=float, default=1.0,
        help="same-capture centroid distance in COLMAP units for consolidating repeat sightings (default: 1.0)",
    )
    compare_matches.add_argument("--output", required=True, help="destination reviewed candidates JSON")
    compare = subparsers.add_parser(
        "compare-observations",
        help="associate Phase 4 tracks conservatively and write candidate change reasoning",
    )
    compare.add_argument("observations", help="Phase 4 observations.json")
    compare.add_argument("--review", help="optional JSON with reviewed identity, visibility, and object anchors")
    compare.add_argument(
        "--movement-threshold",
        type=float,
        help="required map-unit displacement threshold for reviewed object anchors; COLMAP units are arbitrary",
    )
    compare.add_argument("--output", required=True, help="destination candidate report JSON")
    visibility = subparsers.add_parser(
        "review-old-location",
        help="project a reviewed object's old center into after frames and create a manual visibility review gallery",
    )
    visibility.add_argument("candidates", help="reviewed object match candidates JSON")
    visibility.add_argument("annotations", help="object annotation JSON used for the match")
    visibility.add_argument("--shared-model", required=True, help="shared COLMAP binary model")
    visibility.add_argument("--after-images", required=True, help="after RGB frame directory")
    visibility.add_argument("--candidate-id", help="candidate ID; defaults to the first same_object_reviewed candidate")
    visibility.add_argument("--output-dir", required=True, help="destination for review HTML and JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "import-point-cloud":
        try:
            result = import_point_cloud(args.path)
        except (OSError, ValueError, PointCloudFormatError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "point_count": len(result.point_cloud.points),
            "has_colors": result.point_cloud.colors is not None,
            "coordinate_frame": result.coordinate_frame,
            "scale_status": result.scale_status,
            "source": result.metadata["source_path"],
        }))
        return 0
    if args.command == "register-colmap":
        try:
            before = load_colmap_sparse_model(args.before_model, args.before_database)
            after = load_colmap_sparse_model(args.after_model, args.after_database)
            pairs = match_colmap_sparse_models(after, before, ratio=args.descriptor_ratio)
            result = register_point_clouds(
                after.point_cloud,
                before.point_cloud,
                pairs,
                source_frame="after",
                target_frame="before",
                scale_known=args.scale_known,
                inlier_threshold=args.inlier_threshold,
                max_iterations=args.max_iterations,
                icp_iterations=args.icp_iterations,
            )
        except (OSError, ValueError, PointCloudFormatError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        transform = result.transform
        print(json.dumps({
            "success": result.diagnostics.success,
            "before_points": len(before.point_cloud.points),
            "after_points": len(after.point_cloud.points),
            "matched_3d_pairs": len(pairs),
            "diagnostics": asdict(result.diagnostics),
            "transform_after_to_before": None if transform is None else {
                "scale": transform.scale,
                "scale_status": transform.scale_status,
                "rotation": transform.rotation.tolist(),
                "translation": transform.translation.tolist(),
                "matrix": transform.matrix.tolist(),
            },
        }))
        return 0 if result.diagnostics.success else 3
    if args.command == "localize-colmap":
        try:
            result = localize_after_images(
                args.before_model,
                args.before_database,
                args.after_images,
                args.output_dir,
                colmap_executable=args.colmap,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["localization_success"] else 3
    if args.command == "reconstruct-sparse-room":
        try:
            result = reconstruct_sparse_room(
                args.shared_model,
                args.output_dir,
                object_report=args.object_report,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "reconstruct-openmvs-cpu":
        try:
            result = reconstruct_openmvs_cpu(
                args.shared_model,
                args.before_images,
                args.after_images,
                args.object_report,
                args.openmvs_bin,
                args.output_dir,
                max_threads=args.max_threads,
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "observations":
        try:
            report = build_observation_report(
                args.annotations,
                args.shared_model,
                args.before_images,
                args.after_images,
                frame_manifest=args.frame_manifest,
            )
            output = write_observation_report(report, args.output)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "observations": report["observation_count"],
            "tracks": report["track_count"],
            "output": str(output),
        }, indent=2))
        return 0
    if args.command == "detect-objects":
        try:
            image_dirs = {"before": args.before_images, "after": args.after_images}
            selected: dict[str, list[str]] = {"before": [], "after": []}
            for item in args.frame:
                capture, separator, filename = item.partition(":")
                if (
                    not separator
                    or capture not in selected
                    or not filename
                    or Path(filename).name != filename
                ):
                    raise ValueError(f"invalid --frame {item!r}; use before:FILENAME or after:FILENAME")
                frame_path = Path(image_dirs[capture]) / filename
                if not frame_path.is_file():
                    raise FileNotFoundError(f"selected {capture} frame does not exist: {frame_path}")
                selected[capture].append(str(frame_path))
            detector = UltralyticsSegDetector(
                args.model,
                confidence=args.conf,
                image_size=args.imgsz,
                device=args.device,
                preview_dir=args.preview_dir,
            )
            annotations = annotations_from_detector(detector, selected)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(annotations, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "annotations": len(annotations["observations"]),
            "frames": sum(len(frames) for frames in selected.values()),
            "output": str(output.resolve()),
            "previews": None if args.preview_dir is None else str(Path(args.preview_dir).resolve()),
            "device": detector.device,
        }, indent=2))
        return 0
    if args.command == "estimate-depth":
        try:
            image_dirs = {"before": args.before_images, "after": args.after_images}
            selected: list[tuple[str, Path]] = []
            for item in args.frame:
                capture, separator, filename = item.partition(":")
                if (
                    not separator
                    or capture not in image_dirs
                    or not filename
                    or Path(filename).name != filename
                ):
                    raise ValueError(f"invalid --frame {item!r}; use before:FILENAME or after:FILENAME")
                frame_path = Path(image_dirs[capture]) / filename
                if not frame_path.is_file():
                    raise FileNotFoundError(f"selected {capture} frame does not exist: {frame_path}")
                selected.append((capture, frame_path))

            image_anchors = read_colmap_depth_anchors(args.shared_model)
            estimator = DepthAnythingV2SmallEstimator(
                device=args.device, model_path=args.model_path
            )
            output_dir = Path(args.output_dir)
            records: list[dict[str, object]] = []
            for capture, frame_path in selected:
                relative_depth = estimator.predict(frame_path)
                stem = f"{capture}_{frame_path.stem}"
                raw_path = output_dir / "relative" / f"{stem}.npy"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, relative_depth)
                write_depth_preview(relative_depth, output_dir / "previews" / f"{stem}.png")

                registered_name, anchors = anchors_for_frame(
                    image_anchors, frame_path.name, capture
                )
                calibration = calibrate_relative_depth(relative_depth, anchors)
                calibrated_path = None
                if calibration.status == "calibrated":
                    camera_z = relative_depth * float(calibration.scale) + float(calibration.offset)
                    camera_z[camera_z <= 0] = np.nan
                    calibrated_path_obj = output_dir / "camera_z_map_units" / f"{stem}.npy"
                    calibrated_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    np.save(calibrated_path_obj, camera_z)
                    write_depth_preview(
                        camera_z, output_dir / "calibrated_previews" / f"{stem}.png"
                    )
                    calibrated_path = str(calibrated_path_obj.resolve())
                records.append({
                    "capture": capture,
                    "frame": frame_path.name,
                    "registered_image_name": registered_name,
                    "relative_depth_npy": str(raw_path.resolve()),
                    "calibrated_camera_z_npy": calibrated_path,
                    "calibration": calibration_as_dict(calibration),
                })

            report = {
                "schema_version": 1,
                "model": estimator.model_source,
                "device": estimator.device,
                "coordinate_frame": "camera_z_in_before_colmap_map_units",
                "metric_scale": False,
                "calibration_method": "robust affine fit with held-out COLMAP point observations",
                "rejected_fit_policy": "relative-depth visualization only; no calibrated depth or 3D back-projection",
                "frames": records,
            }
            output = write_depth_report(report, output_dir / "depth_calibration.json")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "frames": len(records),
            "calibrated": sum(
                item["calibration"]["status"] == "calibrated" for item in records
            ),
            "output": str(output.resolve()),
            "device": estimator.device,
        }, indent=2))
        return 0
    if args.command == "backproject-objects":
        try:
            if args.max_points_per_mask < 1 or args.minimum_points_per_mask < 1:
                raise ValueError("point limits must be positive")
            report = backproject_object_masks(
                args.annotations,
                args.depth_report,
                args.shared_model,
                args.before_images,
                args.after_images,
                args.output_dir,
                max_points_per_mask=args.max_points_per_mask,
                minimum_points_per_mask=args.minimum_points_per_mask,
                triangulate_labels=tuple(args.triangulate_label),
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "observations": report["observation_count"],
            "backprojected": report["backprojected_count"],
            "triangulated": report["triangulated_count"],
            "image_space_only": report["image_space_only_count"],
            "points": report["point_count"],
            "point_cloud": report["point_cloud"],
            "report": str(Path(args.output_dir, "object_geometry.json").resolve()),
        }, indent=2))
        return 0
    if args.command == "rank-object-matches":
        try:
            result = rank_object_matches(
                args.geometry,
                args.before_images,
                args.after_images,
                args.output_dir,
                model_name_or_path=args.model,
                device=args.device,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "before_objects": result["before_object_count"],
            "after_objects": result["after_object_count"],
            "ranked_before_objects": len(result["rankings"]),
            "json": str(Path(args.output_dir, "dino_match_candidates.json").resolve()),
            "review_page": str(Path(args.output_dir, "dino_match_review.html").resolve()),
            "device": result["device"],
        }, indent=2))
        return 0
    if args.command == "compare-reviewed-matches":
        try:
            result = compare_reviewed_object_matches(
                args.geometry,
                args.rankings,
                args.review,
                args.output,
                movement_threshold=args.movement_threshold,
                track_distance_threshold=args.track_distance_threshold,
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "candidates": result["candidate_count"],
            "output": str(Path(args.output).resolve()),
            "preview": result.get("preview"),
            "movement_threshold_map_units": result["movement_threshold_colmap_units"],
        }, indent=2))
        return 0
    if args.command == "compare-observations":
        try:
            report = compare_observation_reports(
                args.observations,
                review_path=args.review,
                movement_threshold=args.movement_threshold,
            )
            output = write_change_report(report, args.output)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "candidates": report["candidate_count"],
            "output": str(output),
        }, indent=2))
        return 0
    if args.command == "review-old-location":
        try:
            result = build_visibility_evidence(
                args.candidates, args.annotations, args.shared_model, args.after_images,
                args.output_dir, candidate_id=args.candidate_id,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
        return 0
    try:
        manifest = preprocess_pair(
            args.before,
            args.after,
            args.output_dir,
            sample_interval_seconds=args.interval,
            max_frames=args.max_frames,
        )
    except (OSError, ValueError, VideoValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest": f"{args.output_dir}/frame_manifest.json",
                "before_frames": len(manifest["videos"]["before"]["frames"]),
                "after_frames": len(manifest["videos"]["after"]["frames"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
