"""DINOv2 appearance rankings for cross-capture object-mask review."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DINO_MODEL = "facebook/dinov2-small"


def _read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read object geometry report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("object geometry report must have schema_version: 1")
    if not isinstance(report.get("observations"), list):
        raise ValueError("object geometry report must contain observations")
    return report


def _track_geometry_quality(members: list[dict[str, Any]], track_center: list[float], distance_limit: float) -> dict[str, Any]:
    """Summarize simple, explicit quality gates for provisional mask geometry."""
    scores = [float(item["score"]) for item in members if isinstance(item.get("score"), (int, float))]
    holdout = [
        float(item["depth_calibration"]["holdout_p90_relative_error"])
        for item in members
        if isinstance(item.get("depth_calibration"), dict)
        and isinstance(item["depth_calibration"].get("holdout_p90_relative_error"), (int, float))
    ]
    center = np.asarray(track_center, dtype=np.float64)
    spread = [float(np.linalg.norm(np.asarray(item["centroid_xyz_colmap"], dtype=np.float64) - center)) for item in members]
    if all(item.get("geometry_method") == "registered_mask_centroid_ray_intersection" for item in members):
        triangulations = [item.get("triangulation", {}) for item in members]
        view_count = min((int(item.get("view_count", 0)) for item in triangulations), default=0)
        p90 = max((float(item.get("p90_ray_residual_colmap_units", float("inf"))) for item in triangulations), default=float("inf"))
        max_spread = max(spread, default=0.0)
        failures = []
        if not scores or min(scores) < 0.40:
            failures.append("detector confidence below 0.40 or unavailable")
        if view_count < 4 or p90 > 0.48:
            failures.append("multiview ray triangulation has too few views or excessive residual")
        if max_spread > distance_limit:
            failures.append("within-track triangulated centers are inconsistent")
        return {
            "geometry_method": "registered_mask_centroid_ray_intersection",
            "minimum_detector_confidence": min(scores) if scores else None,
            "minimum_detector_confidence_required": 0.40,
            "minimum_triangulation_view_count": view_count,
            "maximum_ray_p90_residual_colmap_units": p90,
            "maximum_ray_p90_residual_allowed_colmap_units": 0.48,
            "maximum_member_centroid_deviation_colmap_units": max_spread,
            "track_distance_limit_colmap_units": distance_limit,
            "multi_frame_geometry": len({item.get("frame") for item in members}) > 1,
            "status": "failed" if failures else "passed",
            "failures": failures,
        }
    checks = {
        "minimum_detector_confidence": min(scores) if scores else None,
        "minimum_detector_confidence_required": 0.40,
        "maximum_depth_holdout_p90_relative_error": max(holdout) if holdout else None,
        "maximum_depth_holdout_error_allowed": 0.10,
        "maximum_member_centroid_deviation_colmap_units": max(spread, default=0.0),
        "track_distance_limit_colmap_units": distance_limit,
        "point_counts": [int(item.get("point_count", 0)) for item in members],
        "multi_frame_geometry": len({item.get("frame") for item in members}) > 1,
    }
    failures = []
    if not scores or min(scores) < checks["minimum_detector_confidence_required"]:
        failures.append("detector confidence below 0.40 or unavailable")
    if not holdout or max(holdout) > checks["maximum_depth_holdout_error_allowed"]:
        failures.append("depth calibration holdout p90 exceeds 10% or is unavailable")
    if max(spread, default=0.0) > distance_limit:
        failures.append("within-track 3D centroids are inconsistent")
    if not checks["multi_frame_geometry"]:
        checks["caveat"] = "single-frame track; temporal geometry consistency cannot be checked"
    checks["status"] = "failed" if failures else ("passed_single_frame_checks" if not checks["multi_frame_geometry"] else "passed")
    checks["failures"] = failures
    return checks


def write_object_change_preview(report_path: str | Path, point_cloud_path: str | Path, output_path: str | Path) -> Path:
    """Write a dependency-free, rotatable 3D canvas preview of object points and associations."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    points_path = Path(point_cloud_path)
    with points_path.open("r", encoding="utf-8") as stream:
        line = stream.readline()
        if line.strip() != "ply":
            raise ValueError("object preview currently requires an ASCII PLY point cloud")
        properties: list[str] = []
        vertex_count = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid PLY header: missing end_header")
            fields = line.split()
            if len(fields) >= 3 and fields[:2] == ["format", "ascii"]:
                pass
            elif len(fields) == 3 and fields[0] == "element" and fields[1] == "vertex":
                vertex_count = int(fields[2])
            elif len(fields) == 3 and fields[0] == "property" and vertex_count is not None:
                properties.append(fields[2])
            elif fields and fields[0] == "end_header":
                break
        if vertex_count is None or not {"x", "y", "z", "object_index", "capture_id"}.issubset(properties):
            raise ValueError("PLY needs x/y/z, object_index, and capture_id vertex properties")
        column = {key: properties.index(key) for key in ("x", "y", "z", "object_index", "capture_id")}
        step = max(1, vertex_count // 9000)
        points = []
        for index, line in enumerate(stream):
            if index % step != 0:
                continue
            values = line.split()
            points.append([float(values[column[k]]) for k in ("x", "y", "z")] + [int(values[column["capture_id"]])])
    tracks = report.get("tracking", {}).get("tracks", [])
    candidates = report.get("candidates", [])
    data = {"points": points, "tracks": tracks, "candidates": candidates}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = r"""<!doctype html><meta charset="utf-8"><title>Object change map</title>
<style>body{font:15px system-ui;margin:1.2rem;color:#172033}canvas{width:min(100%,1100px);height:650px;border:1px solid #ccd2dc;background:#f8fafc;touch-action:none}li{margin:.35rem 0}.hint{color:#526074}</style>
<h1>Object tracks in the shared sparse map</h1><p class="hint">Drag to rotate; scroll to zoom. Before points are blue, after points orange. Green links are reviewed same-object candidates. Coordinates and distances are arbitrary COLMAP map units, not meters.</p>
<canvas id="map" width="1100" height="650"></canvas><h2>Reviewed track comparisons</h2><ul id="list"></ul>
<script>const D=__DATA__, c=document.getElementById('map'),x=c.getContext('2d');let yaw=.65,pitch=.35,scale=15,drag=null;
const all=[...D.points.map(p=>p.slice(0,3)),...D.tracks.map(t=>t.centroid_xyz_colmap)];
const mins=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),maxs=[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))),mid=mins.map((v,k)=>(v+maxs[k])/2);
function project(p){let X=p[0]-mid[0],Y=p[1]-mid[1],Z=p[2]-mid[2],a=X*Math.cos(yaw)-Y*Math.sin(yaw),b=X*Math.sin(yaw)+Y*Math.cos(yaw),v=Z*Math.cos(pitch)-b*Math.sin(pitch);return [c.width/2+a*scale,c.height/2-v*scale]}
function draw(){x.clearRect(0,0,c.width,c.height);for(const p of D.points){const q=project(p);x.fillStyle=p[3]===0?'rgba(45,100,210,.32)':'rgba(235,130,40,.32)';x.fillRect(q[0],q[1],2,2)}
for(const z of D.candidates){if(z.candidate_type!=='same_object_reviewed')continue;const a=D.tracks.find(t=>t.track_id===z.before.track_id),b=D.tracks.find(t=>t.track_id===z.after.track_id);if(!a||!b)continue;const A=project(a.centroid_xyz_colmap),B=project(b.centroid_xyz_colmap);x.strokeStyle='#138a55';x.lineWidth=2;x.beginPath();x.moveTo(...A);x.lineTo(...B);x.stroke()}
for(const t of D.tracks){const q=project(t.centroid_xyz_colmap);x.beginPath();x.arc(q[0],q[1],7,0,Math.PI*2);x.fillStyle=t.capture==='before'?'#2864d2':'#e78328';x.fill();x.fillStyle='#172033';x.font='12px system-ui';x.fillText(t.label+' ['+t.object_indices.join(',')+']',q[0]+9,q[1]-6)}}draw();
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};c.onwheel=e=>{scale=Math.max(3,Math.min(100,scale*(e.deltaY<0?1.1:.9)));draw();e.preventDefault()};
document.getElementById('list').innerHTML=D.candidates.map(z=>'<li><b>'+z.candidate_type+'</b>: '+z.label+'; before '+z.before.object_indices+' → after '+z.after.object_indices+'; displacement '+z.centroid_displacement_colmap_units.toFixed(3)+' map units; '+z.reason+'</li>').join('');</script>""".replace("__DATA__", payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def _masked_crop(item: dict[str, Any], image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read object crop source image: {image_path}")
    height, width = image.shape[:2]
    box = item.get("bbox_xyxy")
    polygon = item.get("polygon_xy")
    if not isinstance(box, list) or len(box) != 4 or not isinstance(polygon, list) or len(polygon) < 3:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2].copy()
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None
    points[:, 0] -= x1
    points[:, 1] -= y1
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    # Neutral fill avoids letting pixels outside the object dominate the crop embedding.
    crop[mask == 0] = (123, 116, 103)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _save_review_page(
    path: Path,
    rankings: list[dict[str, Any]],
    crops: dict[int, str],
) -> None:
    rows: list[str] = []
    for group in rankings:
        before = group["before"]
        for pair in group["ranked_after"]:
            after = pair["after"]
            before_src = html.escape(Path(crops[int(before["object_index"])]).relative_to(path.parent.resolve()).as_posix())
            after_src = html.escape(Path(crops[int(after["object_index"])]).relative_to(path.parent.resolve()).as_posix())
            rows.append(
                f"<tr data-before-index='{int(before['object_index'])}' data-after-index='{int(after['object_index'])}' "
                f"data-similarity='{pair['cosine_similarity']:.8f}'>"
                f"<td><img src='{before_src}'><br>{html.escape(str(before['capture']))}: "
                f"{html.escape(str(before['frame']))}<br>{html.escape(str(before['label']))}</td>"
                f"<td><img src='{after_src}'><br>{html.escape(str(after['capture']))}: "
                f"{html.escape(str(after['frame']))}<br>{html.escape(str(after['label']))}</td>"
                f"<td>{pair['cosine_similarity']:.4f}</td>"
                f"<td>{'same label' if pair['same_label'] else 'label differs'}</td>"
                "<td><select class='same'><option value='unknown' selected>Unreviewed</option>"
                "<option value='true'>Same object</option><option value='false'>Different objects</option></select></td>"
                "<td><select class='geometry'><option value='unknown' selected>Unreviewed</option>"
                "<option value='true'>Geometry acceptable</option><option value='false'>Reject geometry</option></select></td>"
                "</tr>"
            )
    document = """<!doctype html><html><head><meta charset="utf-8">
<title>DINOv2 object match review</title><style>
body{font:16px system-ui,sans-serif;margin:2rem;color:#172033}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccd2dc;padding:.7rem;text-align:left;vertical-align:top}
th{background:#edf1f7}img{width:180px;height:180px;object-fit:contain;background:#eee}
</style></head><body><h1>Appearance match suggestions</h1>
<p>Sorted by DINOv2 cosine similarity. Scores suggest visual resemblance and do not establish identity.
Review each crop. Only accept geometry if the mask and 3D points look usable.</p>
<button id="download">Download review JSON</button>
<table><thead><tr><th>Before mask</th><th>After mask</th><th>Cosine similarity</th><th>Label check</th><th>Identity review</th><th>Geometry review</th></tr></thead>
<tbody>""" + "\n".join(rows) + "</tbody></table></body></html>"
    document = document.replace("</body>", r"""
<script>
document.getElementById('download').onclick=()=>{
 const pairs=[...document.querySelectorAll('tbody tr')].map(row=>({
  before_object_index:Number(row.dataset.beforeIndex),
  after_object_index:Number(row.dataset.afterIndex),
  cosine_similarity:Number(row.dataset.similarity),
  same_object:row.querySelector('.same').value==='unknown'?'unknown':row.querySelector('.same').value==='true',
  geometry_accepted:row.querySelector('.geometry').value==='unknown'?'unknown':row.querySelector('.geometry').value==='true'
 }));
 const blob=new Blob([JSON.stringify({schema_version:1,pairs},null,2)+'\n'],{type:'application/json'});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='dino_match_review.json';link.click();URL.revokeObjectURL(link.href);
};
</script></body>""", 1)
    path.write_text(document, encoding="utf-8")


def rank_object_matches(
    geometry_path: str | Path,
    before_images: str | Path,
    after_images: str | Path,
    output_dir: str | Path,
    *,
    model_name_or_path: str = DEFAULT_DINO_MODEL,
    device: str = "auto",
    top_k: int = 5,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Rank every after-capture mask crop for each before mask using cosine similarity."""
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    geometry_path, output_dir = Path(geometry_path), Path(output_dir)
    report = _read_report(geometry_path)
    observations = [
        item for item in report["observations"]
        if isinstance(item, dict) and item.get("geometry_status") in {
            "backprojected_provisional", "triangulated_provisional"
        }
    ]
    before = [item for item in observations if item.get("capture") == "before"]
    after = [item for item in observations if item.get("capture") == "after"]
    if not before or not after:
        raise ValueError("object geometry report needs back-projected masks from both captures")

    image_dirs = {"before": Path(before_images), "after": Path(after_images)}
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_arrays: list[np.ndarray] = []
    crop_paths: dict[int, str] = {}
    crop_items: list[dict[str, Any]] = []
    for item in observations:
        capture, frame = item.get("capture"), item.get("frame")
        if capture not in image_dirs or not isinstance(frame, str):
            continue
        object_index = int(item.get("object_index", len(crop_items)))
        crop = _masked_crop(item, image_dirs[capture] / frame)
        if crop is None:
            item["appearance_status"] = "skipped_missing_valid_polygon"
            continue
        crop_path = crop_dir / f"object_{object_index:04d}.png"
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(crop_path), bgr):
            raise OSError(f"could not save review crop {crop_path}")
        crop_arrays.append(crop)
        crop_paths[object_index] = str(crop_path.resolve())
        crop_items.append(item)

    usable_before = [item for item in before if int(item.get("object_index", -1)) in crop_paths]
    usable_after = [item for item in after if int(item.get("object_index", -1)) in crop_paths]
    if not usable_before or not usable_after:
        raise ValueError("no usable polygon crops remain in both captures")

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("DINOv2 ranking requires the optional 'depth' Transformers dependency") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment cannot access a CUDA device")
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device).eval()

    embeddings: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(crop_arrays), batch_size):
            batch = processor(images=crop_arrays[start:start + batch_size], return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            features = model(**batch).last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1).cpu().numpy()
            for offset, vector in enumerate(features):
                item = crop_items[start + offset]
                object_index = int(item.get("object_index", -1))
                embeddings[object_index] = vector
                item["appearance_status"] = "embedded"

    rankings: list[dict[str, Any]] = []
    after_rows = [item for item in usable_after if int(item["object_index"]) in embeddings]
    for before_item in usable_before:
        before_index = int(before_item["object_index"])
        if before_index not in embeddings:
            continue
        candidates: list[dict[str, Any]] = []
        for after_item in after_rows:
            after_index = int(after_item["object_index"])
            similarity = float(np.dot(embeddings[before_index], embeddings[after_index]))
            candidates.append({
                "after": {
                    "object_index": after_index,
                    "capture": after_item.get("capture"),
                    "frame": after_item.get("frame"),
                    "label": after_item.get("label"),
                    "score": after_item.get("score"),
                    "centroid_xyz_colmap": after_item.get("centroid_xyz_colmap"),
                    "crop": Path(crop_paths[after_index]).relative_to(output_dir.resolve()).as_posix(),
                },
                "cosine_similarity": similarity,
                "same_label": str(before_item.get("label", "")).casefold() == str(after_item.get("label", "")).casefold(),
                "identity_review": "unreviewed",
            })
        candidates.sort(key=lambda item: item["cosine_similarity"], reverse=True)
        rankings.append({
            "before": {
                "object_index": before_index,
                "capture": before_item.get("capture"),
                "frame": before_item.get("frame"),
                "label": before_item.get("label"),
                "score": before_item.get("score"),
                "centroid_xyz_colmap": before_item.get("centroid_xyz_colmap"),
                "crop": Path(crop_paths[before_index]).relative_to(output_dir.resolve()).as_posix(),
            },
            "ranked_after": candidates[:top_k],
        })

    result: dict[str, Any] = {
        "schema_version": 1,
        "model": model_name_or_path,
        "device": device,
        "coordinate_frame": "before_colmap_map",
        "metric_scale": False,
        "similarity": "cosine similarity of L2-normalized DINOv2 CLS embeddings from polygon-masked object crops",
        "decision_policy": "rankings are suggestions only; no cross-capture identity is assigned",
        "before_object_count": len(usable_before),
        "after_object_count": len(usable_after),
        "rankings": rankings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dino_match_candidates.json"
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    result["output"] = str(output_path.resolve())
    _save_review_page(output_dir / "dino_match_review.html", rankings, crop_paths)
    return result


def compare_reviewed_object_matches(
    geometry_path: str | Path,
    ranking_path: str | Path,
    review_path: str | Path,
    output_path: str | Path,
    *,
    movement_threshold: float | None = None,
    track_distance_threshold: float = 1.0,
) -> dict[str, Any]:
    """Consolidate same-capture sightings, then compare reviewed cross-capture tracks."""
    if movement_threshold is not None and (not np.isfinite(movement_threshold) or movement_threshold <= 0):
        raise ValueError("movement_threshold must be finite and greater than zero")
    if not np.isfinite(track_distance_threshold) or track_distance_threshold <= 0:
        raise ValueError("track_distance_threshold must be finite and greater than zero")
    geometry = _read_report(Path(geometry_path))
    try:
        rankings = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read DINO ranking report {ranking_path}: {exc}") from exc
    if not isinstance(rankings, dict) or rankings.get("schema_version") != 1 or not isinstance(rankings.get("rankings"), list):
        raise ValueError("DINO ranking file must contain schema_version: 1 and a rankings array")
    try:
        review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read identity review {review_path}: {exc}") from exc
    if not isinstance(review, dict) or review.get("schema_version") != 1 or not isinstance(review.get("pairs"), list):
        raise ValueError("review file must contain schema_version: 1 and a pairs array")

    objects = {
        int(item["object_index"]): item
        for item in geometry["observations"]
        if isinstance(item, dict) and item.get("geometry_status") in {
            "backprojected_provisional", "triangulated_provisional"
        }
        and isinstance(item.get("object_index"), int)
    }

    # Associate repeat detections across frames in each capture. Tracks are
    # conservative: labels must agree, map-space centroids must be close, and
    # a track can contain at most one detection from any frame. Complete-link
    # distance checks prevent chained nearby objects from collapsing together.
    by_capture: dict[str, list[dict[str, Any]]] = {"before": [], "after": []}
    for obj in objects.values():
        by_capture.setdefault(str(obj.get("capture")), []).append(obj)
    track_for_object: dict[int, int] = {}
    tracks: list[dict[str, Any]] = []
    for capture, capture_objects in by_capture.items():
        capture_objects.sort(key=lambda obj: (str(obj.get("frame", "")), int(obj["object_index"])))
        clusters: list[list[dict[str, Any]]] = []
        for obj in capture_objects:
            xyz = np.asarray(obj["centroid_xyz_colmap"], dtype=np.float64)
            options: list[tuple[float, int]] = []
            for cluster_idx, cluster in enumerate(clusters):
                if any(member.get("frame") == obj.get("frame") for member in cluster):
                    continue
                if any(str(member.get("label", "")).casefold() != str(obj.get("label", "")).casefold() for member in cluster):
                    continue
                distances = [float(np.linalg.norm(xyz - np.asarray(member["centroid_xyz_colmap"], dtype=np.float64))) for member in cluster]
                if max(distances, default=float("inf")) <= track_distance_threshold:
                    options.append((float(np.mean(distances)), cluster_idx))
            if options:
                clusters[min(options)[1]].append(obj)
            else:
                clusters.append([obj])
        for cluster in clusters:
            track_id = len(tracks)
            for obj in cluster:
                track_for_object[int(obj["object_index"])] = track_id
            xyzs = np.asarray([obj["centroid_xyz_colmap"] for obj in cluster], dtype=np.float64)
            tracks.append({
                "track_id": track_id,
                "capture": capture,
                "label": cluster[0].get("label"),
                "object_indices": [int(obj["object_index"]) for obj in cluster],
                "frames": [obj.get("frame") for obj in cluster],
                "centroid_xyz_colmap": np.median(xyzs, axis=0).tolist(),
                "geometry_quality": _track_geometry_quality(cluster, np.median(xyzs, axis=0).tolist(), track_distance_threshold),
            })
    offered: set[tuple[int, int]] = set()
    for group in rankings.get("rankings", []):
        if not isinstance(group, dict):
            continue
        before_id = int(group["before"]["object_index"])
        for pair in group.get("ranked_after", []):
            offered.add((before_id, int(pair["after"]["object_index"])))

    reviewed_pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for item in review["pairs"]:
        if not isinstance(item, dict):
            raise ValueError("each reviewed pair must be an object")
        before_id, after_id = item.get("before_object_index"), item.get("after_object_index")
        if not isinstance(before_id, int) or not isinstance(after_id, int):
            raise ValueError("review pair object indices must be integers")
        pair_key = before_id, after_id
        if pair_key in seen:
            raise ValueError(f"duplicate reviewed pair {pair_key}")
        seen.add(pair_key)
        if pair_key not in offered:
            raise ValueError(f"review pair {pair_key} was not offered by the DINO ranking")
        before, after = objects.get(before_id), objects.get(after_id)
        if before is None or after is None or before.get("capture") != "before" or after.get("capture") != "after":
            raise ValueError(f"review pair {pair_key} does not identify before/after geometry")
        same = item.get("same_object", "unknown")
        geometry_ok = item.get("geometry_accepted", "unknown")
        if same not in {True, False, "unknown"}:
            raise ValueError("same_object must be true, false, or 'unknown'")
        if geometry_ok not in {True, False, "unknown"}:
            raise ValueError("geometry_accepted must be true, false, or 'unknown'")
        before_track = track_for_object[before_id]
        after_track = track_for_object[after_id]
        reviewed_pairs.append({
            "before_track_id": before_track,
            "after_track_id": after_track,
            "before_object_index": before_id,
            "after_object_index": after_id,
            "same_object": same,
            "geometry_accepted": geometry_ok,
            "cosine_similarity": item.get("cosine_similarity"),
        })

    # Roll reviewed mask-pair decisions up to track-pair decisions so that an
    # object seen in several frames is reported once.
    pair_decisions: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for pair in reviewed_pairs:
        pair_decisions.setdefault((pair["before_track_id"], pair["after_track_id"]), []).append(pair)
    positive_pairs = {
        key for key, decisions in pair_decisions.items()
        if any(d["same_object"] is True and d["geometry_accepted"] is True for d in decisions)
    }
    before_degree: dict[int, int] = {}
    after_degree: dict[int, int] = {}
    for before_id, after_id in positive_pairs:
        before_degree[before_id] = before_degree.get(before_id, 0) + 1
        after_degree[after_id] = after_degree.get(after_id, 0) + 1
    results: list[dict[str, Any]] = []
    track_by_id = {track["track_id"]: track for track in tracks}
    for (before_track_id, after_track_id), decisions in sorted(pair_decisions.items()):
        before_track, after_track = track_by_id[before_track_id], track_by_id[after_track_id]
        before_xyz = np.asarray(before_track["centroid_xyz_colmap"], dtype=np.float64)
        after_xyz = np.asarray(after_track["centroid_xyz_colmap"], dtype=np.float64)
        distance = float(np.linalg.norm(after_xyz - before_xyz))
        positives = [d for d in decisions if d["same_object"] is True and d["geometry_accepted"] is True]
        negatives = [d for d in decisions if d["same_object"] is False]
        auto_quality = {
            "before": before_track["geometry_quality"],
            "after": after_track["geometry_quality"],
        }
        quality_passed = all(value["status"] != "failed" for value in auto_quality.values())
        one_to_one = before_degree.get(before_track_id, 0) <= 1 and after_degree.get(after_track_id, 0) <= 1
        if positives:
            same, geometry_ok = True, quality_passed and one_to_one
        elif negatives:
            same, geometry_ok = False, "unknown"
        else:
            same, geometry_ok = "unknown", "unknown"
        if same is False:
            candidate_type, reason = "different_objects", "reviewer marked the instances as different"
        elif same is not True:
            candidate_type, reason = "insufficient_observation", "cross-capture identity is unreviewed"
        elif geometry_ok is False:
            candidate_type, reason = "insufficient_observation", "geometry review or automatic confidence/calibration quality checks failed"
        elif geometry_ok is not True:
            candidate_type, reason = "insufficient_observation", "mask-derived geometry is unreviewed"
        elif movement_threshold is None:
            candidate_type, reason = "same_object_reviewed", "set an explicit threshold in COLMAP map units to assess displacement"
        elif distance > movement_threshold:
            candidate_type, reason = "candidate_moved", "reviewed identity and geometry exceed the configured map-unit threshold"
        else:
            candidate_type, reason = "observed_both", "reviewed geometry is within the configured map-unit threshold"
        results.append({
            "candidate_id": f"before_track:{before_track_id}->after_track:{after_track_id}",
            "candidate_type": candidate_type,
            "label": before_track.get("label"),
            "appearance_similarity": max((float(d["cosine_similarity"]) for d in decisions if d.get("cosine_similarity") is not None), default=None),
            "before": before_track,
            "after": after_track,
            "review": {
                "same_object": same,
                "geometry_accepted": geometry_ok,
                "human_geometry_accepted": any(d["geometry_accepted"] is True for d in decisions),
                "one_to_one_association": one_to_one,
                "automatic_geometry_quality": auto_quality,
                "supporting_pairs": decisions,
            },
            "centroid_displacement_colmap_units": distance,
            "movement_threshold_colmap_units": movement_threshold,
            "reason": reason,
        })

    result = {
        "schema_version": 1,
        "coordinate_frame": "before_colmap_map",
        "metric_scale": False,
        "tracking": {
            "method": "same label and complete-link 3D centroid proximity; at most one detection per frame",
            "distance_threshold_colmap_units": track_distance_threshold,
            "track_count": len(tracks),
            "tracks": tracks,
        },
        "movement_threshold_colmap_units": movement_threshold,
        "policy": {
            "dino_similarity": "candidate ranking only; never confirms instance identity",
            "movement": "requires reviewed same-object and geometry evidence plus a user threshold; displacement is between consolidated track median centroids",
            "added_or_removed": "not inferred by this command; absence is not proof",
        },
        "candidate_count": len(results),
        "candidates": results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_suffix(".html")
    result["preview"] = str(preview_path.resolve())
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    point_cloud_path = geometry.get("point_cloud")
    if point_cloud_path and Path(point_cloud_path).is_file():
        write_object_change_preview(output_path, point_cloud_path, preview_path)
    return result
