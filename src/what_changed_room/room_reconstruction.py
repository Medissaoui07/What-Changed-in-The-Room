"""Build an inspectable room-structure preview from sparse COLMAP evidence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .shared_map import read_sparse_geometry


def _supported_planes(
    points: np.ndarray,
    *,
    max_planes: int = 6,
    minimum_support: int = 30,
    trials: int = 1400,
) -> list[dict[str, object]]:
    """Find large planar point patches with deterministic RANSAC.

    The output is deliberately called a patch: sparse feature support does not
    establish that the entire rectangle/polygon is a physical room surface.
    """
    if len(points) < 3:
        return []
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    threshold = max(diagonal * 0.008, 1e-5)
    remaining = np.arange(len(points))
    rng = np.random.default_rng(17)
    patches: list[dict[str, object]] = []

    for _ in range(max_planes):
        if len(remaining) < minimum_support:
            break
        cloud = points[remaining]
        best: np.ndarray | None = None
        for _trial in range(trials):
            sample = cloud[rng.choice(len(cloud), size=3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-10:
                continue
            normal /= norm
            distances = np.abs((cloud - sample[0]) @ normal)
            inliers = np.flatnonzero(distances <= threshold)
            if best is None or len(inliers) > len(best):
                best = inliers
        if best is None or len(best) < minimum_support:
            break

        inlier_points = cloud[best]
        centroid = inlier_points.mean(axis=0)
        _u, _s, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vh[-1]
        residual = np.abs((inlier_points - centroid) @ normal)
        refined = residual <= threshold
        inlier_points = inlier_points[refined]
        if len(inlier_points) < minimum_support:
            break
        centroid = inlier_points.mean(axis=0)
        _u, _s, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        axis_u, axis_v = vh[0], vh[1]
        local = np.column_stack(((inlier_points - centroid) @ axis_u, (inlier_points - centroid) @ axis_v))
        hull = cv2.convexHull(local.astype(np.float32)).reshape(-1, 2)
        if len(hull) < 3:
            break
        vertices = centroid + hull[:, :1] * axis_u + hull[:, 1:] * axis_v
        rms = float(np.sqrt(np.mean(((inlier_points - centroid) @ vh[-1]) ** 2)))
        area = float(cv2.contourArea(hull))
        if area < diagonal * diagonal * 0.001:
            break

        patches.append({
            "id": len(patches) + 1,
            "vertices_xyz": vertices.tolist(),
            "normal_xyz": vh[-1].tolist(),
            "support_points": int(len(inlier_points)),
            "support_fraction": float(len(inlier_points) / len(points)),
            "rms_residual_map_units": rms,
            "inlier_threshold_map_units": threshold,
            "area_map_units_squared": area,
            "interpretation": "sparse-supported planar patch; room-surface identity unverified",
        })
        removed = remaining[best]
        keep = np.ones(len(remaining), dtype=bool)
        keep[best] = False
        remaining = remaining[keep]
        # Do not repeatedly emit nearly the same plane when a refit slightly
        # changed its inlier set.
        if len(removed) == 0:
            break
    return patches


def _reviewed_object_pairs(report_path: str | Path | None) -> list[dict[str, object]]:
    if report_path is None:
        return []
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    pairs = []
    for candidate in data.get("candidates", []):
        review = candidate.get("review", {})
        if (
            candidate.get("candidate_type") != "same_object_reviewed"
            or not review.get("same_object")
            or not review.get("geometry_accepted")
            or not review.get("human_geometry_accepted")
        ):
            continue
        before = candidate.get("before", {}).get("centroid_xyz_colmap")
        after = candidate.get("after", {}).get("centroid_xyz_colmap")
        if before is None or after is None:
            continue
        pairs.append({
            "label": str(candidate.get("label", "object")),
            "before_xyz": before,
            "after_xyz": after,
            "displacement_map_units": float(np.linalg.norm(np.asarray(after) - np.asarray(before))),
            "appearance_similarity": candidate.get("appearance_similarity"),
            "evidence_note": "reviewed provisional center anchors from multiview masks; not object surfaces",
        })
    return pairs


def _write_viewer(path: Path, points: np.ndarray, cameras: list[object], planes: list[dict[str, object]], pairs: list[dict[str, object]]) -> None:
    camera_rows = [
        {"name": camera.name, "xyz": list(camera.center),
         "capture": "after" if camera.name.startswith("wcr_after_") else "before"}
        for camera in cameras
    ]
    payload = json.dumps({
        "points": points.tolist(), "cameras": camera_rows,
        "planes": planes, "object_pairs": pairs,
    }, separators=(",", ":")).replace("</", "<\\/")
    page = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sparse room model</title><style>
body{font:15px system-ui,sans-serif;margin:20px;color:#172033;background:#f5f7fa}h1{margin-bottom:4px}
.note{max-width:1000px;color:#4b5870}canvas{display:block;width:min(100%,1100px);height:auto;background:#fff;border:1px solid #cbd3df;touch-action:none}
.legend{display:flex;flex-wrap:wrap;gap:18px;margin:10px 0}.tag{font-size:13px;color:#475569}.cards{display:flex;gap:14px;flex-wrap:wrap}
.card{background:white;border:1px solid #dbe1ea;border-radius:8px;padding:12px;min-width:240px}.warn{color:#9a5b00}
</style><h1>Sparse room structure</h1>
<p class="note">Drag to rotate; scroll to zoom. Gray dots are COLMAP feature points. Translucent polygons are planar patches supported by those points, not verified walls or floor. Camera dots show capture poses. Object markers are reviewed provisional centers. All coordinates use arbitrary COLMAP map units, not meters.</p>
<canvas id="scene" width="1100" height="700"></canvas><div class="legend"><span>● sparse points</span><span style="color:#2864d2">● before cameras / object center</span><span style="color:#e78328">● after cameras / object center</span><span style="color:#a4b8d2">▱ inferred planar patch</span></div>
<h2>Object evidence</h2><div id="pairs" class="cards"></div>
<script>const D=__DATA__,c=document.getElementById('scene'),ctx=c.getContext('2d');
const positions=[...D.points,...D.cameras.map(a=>a.xyz),...D.planes.flatMap(p=>p.vertices_xyz),...D.object_pairs.flatMap(p=>[p.before_xyz,p.after_xyz])];
const lo=[0,1,2].map(k=>Math.min(...positions.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...positions.map(p=>p[k]))),mid=lo.map((v,k)=>(v+hi[k])/2),extent=Math.max(...hi.map((v,k)=>v-lo[k]),1e-6);
let yaw=.7,pitch=.35,zoom=.82*Math.min(c.width,c.height)/extent,drag=null;
function proj(p){let X=p[0]-mid[0],Y=p[1]-mid[1],Z=p[2]-mid[2],a=X*Math.cos(yaw)-Y*Math.sin(yaw),b=X*Math.sin(yaw)+Y*Math.cos(yaw),v=b*Math.cos(pitch)-Z*Math.sin(pitch),d=b*Math.sin(pitch)+Z*Math.cos(pitch);return [c.width/2+a*zoom,c.height/2-v*zoom,d]}
function polygon(vs,color){const q=vs.map(proj);ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);q.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='rgba(74,96,125,.6)';ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([])}
function marker(p,color,label,r=6){const q=proj(p);ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.fillStyle='#172033';ctx.font='12px system-ui';ctx.fillText(label,q[0]+8,q[1]-7)}
function draw(){ctx.clearRect(0,0,c.width,c.height);for(const p of D.planes)polygon(p.vertices_xyz,'rgba(138,166,199,.19)');
for(const p of D.points){const q=proj(p);ctx.fillStyle='rgba(90,101,116,.55)';ctx.fillRect(q[0],q[1],2.2,2.2)}
for(const a of D.cameras)marker(a.xyz,a.capture==='before'?'#2864d2':'#e78328',a.capture,4);
for(const p of D.object_pairs){let a=proj(p.before_xyz),b=proj(p.after_xyz);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle='#15835b';ctx.lineWidth=3;ctx.stroke();marker(p.before_xyz,'#2864d2',p.label+' before',8);marker(p.after_xyz,'#e78328',p.label+' after',8)} }
draw();c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()};c.onwheel=e=>{zoom=Math.max(2,Math.min(500,zoom*(e.deltaY<0?1.1:.9)));draw();e.preventDefault()};
document.getElementById('pairs').innerHTML=D.object_pairs.length?D.object_pairs.map(p=>'<div class="card"><b>'+p.label+'</b><p>Reviewed provisional anchor separation: '+p.displacement_map_units.toFixed(3)+' COLMAP map units.</p><span class="warn">'+p.evidence_note+'</span></div>').join(''):'<p>No reviewed before/after object anchors supplied.</p>';</script>'''.replace("__DATA__", payload)
    path.write_text(page, encoding="utf-8")


def reconstruct_sparse_room(
    shared_model: str | Path,
    output_dir: str | Path,
    *,
    object_report: str | Path | None = None,
) -> dict[str, object]:
    """Create a room-structure HTML viewer and evidence JSON from a shared model."""
    points, cameras = read_sparse_geometry(shared_model)
    if len(points) < 3:
        raise ValueError("the shared COLMAP model needs at least three sparse points")
    planes = _supported_planes(points)
    pairs = _reviewed_object_pairs(object_report)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "coordinate_frame": "shared_colmap_map",
        "metric_scale": False,
        "sparse_point_count": int(len(points)),
        "camera_count": len(cameras),
        "before_camera_count": sum(not camera.name.startswith("wcr_after_") for camera in cameras),
        "after_camera_count": sum(camera.name.startswith("wcr_after_") for camera in cameras),
        "planar_patch_count": len(planes),
        "planar_patches": planes,
        "reviewed_object_pairs": pairs,
        "limitations": [
            "Sparse feature points do not form complete surfaces or prove occupancy.",
            "Planar patches are inferred from point support; floor/wall identity is not assigned.",
            "Object markers are provisional centers, not object extents or surfaces.",
            "COLMAP scale is arbitrary; coordinates are not meters.",
        ],
    }
    report_path = output / "sparse_room_model.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    preview_path = output / "sparse_room_model.html"
    _write_viewer(preview_path, points, cameras, planes, pairs)
    return {"report": str(report_path.resolve()), "preview": str(preview_path.resolve()), **{
        key: report[key] for key in (
            "sparse_point_count", "camera_count", "before_camera_count",
            "after_camera_count", "planar_patch_count",
        )
    }}
