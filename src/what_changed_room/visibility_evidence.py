"""Auditable image evidence for checking an object's old location after a move."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .object_geometry import _match_model_image, _read_colmap_cameras, _read_colmap_images


def build_visibility_evidence(
    candidates_path: str | Path,
    annotations_path: str | Path,
    shared_model: str | Path,
    after_images: str | Path,
    output_dir: str | Path,
    *,
    candidate_id: str | None = None,
) -> dict[str, object]:
    """Project a reviewed object's old center into registered after frames.

    This deliberately does not infer full-object visibility from a center point.
    It creates a frame-by-frame review aid; removal remains unproven unless a
    person confirms that the object's full footprint is visible and empty.
    """
    candidates_data = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    annotations_data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    candidates = candidates_data.get("candidates", [])
    eligible = [
        c for c in candidates
        if c.get("candidate_type") in {"same_object_reviewed", "unmatched_before"}
    ]
    candidate = next((c for c in eligible if c.get("candidate_id") == candidate_id), None) if candidate_id else (eligible[0] if eligible else None)
    if candidate is None:
        raise ValueError("no reviewed same-object candidate found; provide a candidate ID from the candidates JSON")
    before_xyz = np.asarray(candidate["before"]["centroid_xyz_colmap"], dtype=np.float64)
    after_xyz = (
        np.asarray(candidate["after"]["centroid_xyz_colmap"], dtype=np.float64)
        if isinstance(candidate.get("after"), dict)
        and candidate.get("after", {}).get("centroid_xyz_colmap") is not None
        else None
    )
    label = str(candidate.get("label", "object"))
    cameras = _read_colmap_cameras(Path(shared_model))
    model_images = _read_colmap_images(Path(shared_model))
    images_dir = Path(after_images)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gallery_frames = out / "frames"
    gallery_frames.mkdir(parents=True, exist_ok=True)
    after_observations = [
        o for o in annotations_data.get("observations", [])
        if o.get("capture") == "after" and str(o.get("label", "")).casefold() == label.casefold()
    ]
    by_frame: dict[str, list[dict[str, object]]] = {}
    for observation in after_observations:
        by_frame.setdefault(str(observation.get("frame")), []).append(observation)

    records: list[dict[str, object]] = []
    for frame_path in sorted(images_dir.glob("*")):
        if frame_path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
            continue
        model_name = _match_model_image(model_images, "after", frame_path.name)
        if model_name is None:
            continue
        gallery_image = gallery_frames / frame_path.name
        if not gallery_image.exists():
            shutil.copy2(frame_path, gallery_image)
        pose = model_images[model_name]
        camera = cameras.get(int(pose["camera_id"]))
        if camera is None:
            continue
        rotation = np.asarray(pose["rotation"], dtype=np.float64)
        translation = np.asarray(pose["translation"], dtype=np.float64)
        camera_xyz = rotation @ before_xyz + translation
        projected = None
        inside = False
        projected_new = None
        new_inside = False
        if camera_xyz[2] > 1e-8:
            point, _ = cv2.projectPoints(
                before_xyz.reshape(1, 1, 3), cv2.Rodrigues(rotation)[0], translation,
                np.asarray(camera["matrix"]), np.asarray(camera["distortion"]),
            )
            x, y = (float(v) for v in point.reshape(2))
            projected = [x, y]
            inside = 0 <= x < int(camera["width"]) and 0 <= y < int(camera["height"])
        if after_xyz is not None:
            new_camera_xyz = rotation @ after_xyz + translation
            if new_camera_xyz[2] > 1e-8:
                new_point, _ = cv2.projectPoints(
                    after_xyz.reshape(1, 1, 3), cv2.Rodrigues(rotation)[0], translation,
                    np.asarray(camera["matrix"]), np.asarray(camera["distortion"]),
                )
                nx, ny = (float(v) for v in new_point.reshape(2))
                projected_new = [nx, ny]
                new_inside = 0 <= nx < int(camera["width"]) and 0 <= ny < int(camera["height"])
        matches = by_frame.get(frame_path.name, [])
        boxes = [m.get("bbox_xyxy") for m in matches if isinstance(m.get("bbox_xyxy"), list)]
        records.append({
            "frame": frame_path.name,
            "image": str(gallery_image.resolve()),
            "image_uri": "frames/" + frame_path.name,
            "image_width": int(camera["width"]),
            "image_height": int(camera["height"]),
            "projected_old_center_xy": projected,
            "old_center_inside_image": inside,
            "projected_new_center_xy": projected_new,
            "new_center_inside_image": new_inside,
            "current_same_class_boxes_xyxy": boxes,
            "footprint_visibility": "unknown",
            "review": "pending",
        })

    report: dict[str, object] = {
        "schema_version": 1,
        "candidate_id": candidate.get("candidate_id"),
        "label": label,
        "finding": (
            "same object is automatically matched across captures; paired old/new centers are projected into after views"
            if candidate.get("candidate_type") == "same_object_reviewed"
            else "before object has no after correspondence; removal is not proven"
        ),
        "removal_status": "not_proven",
        "old_anchor_colmap_units": before_xyz.tolist(),
        "policy": "An in-frame projected center is not proof that the full object footprint is visible. Missing detections or sparse points never prove removal.",
        "frames": records,
    }
    json_path = out / "visibility_evidence.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cards = []
    for row in records:
        img = html.escape(str(row["image_uri"]))
        frame = html.escape(str(row["frame"]))
        width, height = int(row["image_width"]), int(row["image_height"])
        point = row["projected_old_center_xy"]
        marker = ""
        if point is not None:
            marker = f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="9" fill="none" stroke="#ff2d55" stroke-width="3"/><text x="{point[0]+10:.2f}" y="{point[1]:.2f}" fill="#ff2d55" font-size="16">old center</text>'
        new_point = row["projected_new_center_xy"]
        if new_point is not None:
            marker += f'<circle cx="{new_point[0]:.2f}" cy="{new_point[1]:.2f}" r="8" fill="none" stroke="#16a34a" stroke-width="3"/><text x="{new_point[0]+10:.2f}" y="{new_point[1]:.2f}" fill="#16a34a" font-size="16">new center</text>'
        boxes = "".join(
            f'<rect x="{b[0]}" y="{b[1]}" width="{float(b[2])-float(b[0])}" height="{float(b[3])-float(b[1])}" fill="none" stroke="#16a34a" stroke-width="3"/>'
            for b in row["current_same_class_boxes_xyxy"] if len(b) == 4
        )
        view_status = "in frame" if row["old_center_inside_image"] else "outside / unavailable"
        cards.append(
            f'<article><h2>{frame} · old center {view_status}</h2>'
            f'<div class="photo"><img src="{img}" alt="{frame}"><svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">{marker}{boxes}</svg></div>'
            f'<label>Can you see the entire old object footprint clearly? <select data-frame="{frame}"><option value="pending">Choose…</option><option value="fully_visible_empty">Yes, fully visible and empty</option><option value="occluded">No, partly occluded</option><option value="out_of_view">No, not in view</option><option value="uncertain">Uncertain</option></select></label></article>'
        )
    page = '''<!doctype html><meta charset="utf-8"><title>Projected object centers in after views</title>
<style>body{font:16px system-ui;background:#f4f6f8;color:#17202a;margin:2rem}article{background:white;padding:1rem;margin:1rem 0;border-radius:10px} .photo{position:relative;width:min(900px,100%)}img{display:block;width:100%}.photo svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}select,button{font:inherit;padding:.5rem;margin:.5rem}</style>
<h1>Object evidence in registered after views</h1><p>Red marks the old center and green marks the new center when a track was matched. Green boxes are detector masks. The centers are projected 3D anchors; they do not show the complete object footprint. An absent box does not prove removal.</p><button id="save">Download reviewed visibility decisions</button><main>''' + "\n".join(cards) + '''</main><script>
document.querySelector('#save').onclick=()=>{const d={schema_version:1,candidate_id:''' + json.dumps(candidate.get("candidate_id")) + ''',decisions:[...document.querySelectorAll('select')].map(s=>({frame:s.dataset.frame,footprint_visibility:s.value}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));a.download='visibility_review.json';a.click();URL.revokeObjectURL(a.href)};
</script>'''
    (out / "visibility_evidence.html").write_text(page, encoding="utf-8")
    return {"frames": len(records), "json": str(json_path.resolve()), "html": str((out / "visibility_evidence.html").resolve()), "removal_status": "not_proven"}
