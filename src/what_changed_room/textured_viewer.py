"""Create a dependency-free browser viewer for OpenMVS textured PLY meshes."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import numpy as np


def _read_textured_ply(path: Path) -> tuple[np.ndarray, list[tuple[list[int], list[float]]], Path]:
    with path.open("rb") as stream:
        header: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"truncated PLY header in {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"expected binary_little_endian PLY: {path}")
        vertex_count = int(next(row.split()[-1] for row in header if row.startswith("element vertex ")))
        face_count = int(next(row.split()[-1] for row in header if row.startswith("element face ")))
        texture_name = next(
            row.partition("TextureFile")[2].strip().strip('"')
            for row in header if "TextureFile" in row
        )
        properties: list[str] = []
        reading_vertices = False
        for row in header:
            if row.startswith("element vertex "):
                reading_vertices = True
            elif row.startswith("element face "):
                reading_vertices = False
            elif reading_vertices and row.startswith("property "):
                properties.append(row.split()[-1])
        if not {"x", "y", "z"}.issubset(properties):
            raise ValueError(f"PLY has no XYZ vertex coordinates: {path}")
        vertex_property_count = len(properties)
        # OpenMVS textured exports use float32 vertex properties; retain the
        # property lookup so additional normal fields can be skipped safely.
        raw = np.frombuffer(stream.read(vertex_count * vertex_property_count * 4), dtype="<f4")
        if raw.size != vertex_count * vertex_property_count:
            raise ValueError(f"truncated PLY vertices in {path}")
        table = raw.reshape(vertex_count, vertex_property_count)
        xyz = table[:, [properties.index(axis) for axis in ("x", "y", "z")]].astype(np.float32)
        faces: list[tuple[list[int], list[float]]] = []
        for _ in range(face_count):
            count_raw = stream.read(1)
            if len(count_raw) != 1:
                raise ValueError(f"truncated face list in {path}")
            count = count_raw[0]
            indices_raw = stream.read(count * 4)
            if len(indices_raw) != count * 4:
                raise ValueError(f"truncated face indices in {path}")
            indices = list(struct.unpack("<" + "I" * count, indices_raw))
            uv_count_raw = stream.read(1)
            if len(uv_count_raw) != 1:
                raise ValueError(f"missing face UVs in {path}")
            uv_count = uv_count_raw[0]
            uv_raw = stream.read(uv_count * 4)
            if len(uv_raw) != uv_count * 4:
                raise ValueError(f"truncated face UVs in {path}")
            uvs = list(struct.unpack("<" + "f" * uv_count, uv_raw))
            if len(indices) >= 3 and len(uvs) == count * 2:
                faces.append((indices, uvs))
    texture_path = path.parent / texture_name
    if not texture_path.is_file():
        raise FileNotFoundError(f"OpenMVS texture image is missing: {texture_path}")
    return xyz, faces, texture_path


def _triangles(xyz: np.ndarray, faces: list[tuple[list[int], list[float]]], center: np.ndarray, scale: float) -> tuple[list[float], list[float]]:
    positions: list[float] = []
    uvs_out: list[float] = []
    # Keep the inline browser payload responsive for large OpenMVS meshes.
    stride = max(1, len(faces) // 120_000)
    for indices, face_uvs in faces[::stride]:
        for corner in range(1, len(indices) - 1):
            # OpenMVS stores texcoords per face corner, not per global vertex.
            # A shared vertex can have different UVs on different texture seams.
            for face_corner in (0, corner, corner + 1):
                index = indices[face_corner]
                positions.extend(((xyz[index] - center) / scale).tolist())
                uv_offset = face_corner * 2
                uvs_out.extend(face_uvs[uv_offset:uv_offset + 2])
    return positions, uvs_out


def estimate_colmap_world_up(model_dir: str | Path) -> list[float] | None:
    """Estimate scene up from the registered before-camera image vertical axes."""
    images_path = Path(model_dir) / "images.bin"
    if not images_path.is_file():
        return None
    image_ups: list[np.ndarray] = []
    with images_path.open("rb") as stream:
        count_raw = stream.read(8)
        if len(count_raw) != 8:
            return None
        image_count, = struct.unpack("<Q", count_raw)
        for _ in range(image_count):
            image_id_raw = stream.read(4)
            if len(image_id_raw) != 4:
                return None
            quat_raw = stream.read(32)
            if len(quat_raw) != 32:
                return None
            qw, qx, qy, qz = struct.unpack("<4d", quat_raw)
            if len(stream.read(24)) != 24:  # translation
                return None
            if len(stream.read(4)) != 4:  # camera ID
                return None
            name_bytes = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    return None
                if char == b"\0":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8", errors="replace")
            point_count_raw = stream.read(8)
            if len(point_count_raw) != 8:
                return None
            point_count, = struct.unpack("<Q", point_count_raw)
            stream.seek(point_count * 24, 1)  # x, y, point3D ID
            if not name.startswith("before_"):
                continue
            # COLMAP quaternions map world vectors into camera coordinates.
            # A landscape camera's image-up vector is camera -Y.
            rotation = np.asarray([
                [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
                [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
                [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
            ], dtype=np.float64)
            image_ups.append(rotation.T @ np.asarray([0.0, -1.0, 0.0]))
    if not image_ups:
        return None
    ups = np.asarray(image_ups)
    reference = ups[0]
    ups[np.einsum("ij,j->i", ups, reference) < 0] *= -1
    mean = ups.mean(axis=0)
    length = float(np.linalg.norm(mean))
    if length < 1e-6:
        return None
    mean /= length
    # Discard camera poses that disagree substantially with the dominant
    # landscape orientation (for example, a frame captured while rotating).
    inliers = ups[(ups @ mean) >= 0.75]
    if len(inliers):
        mean = inliers.mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-6)
    return mean.tolist()


def write_textured_comparison(
    before_mesh: str | Path,
    after_mesh: str | Path,
    output_html: str | Path,
    *,
    movements: list[dict[str, object]] | None = None,
    meters_per_map_unit: float | None = None,
    shared_model: str | Path | None = None,
) -> Path:
    """Write an interactive, side-by-side RGB texture viewer with old/new anchors."""
    before_xyz, before_faces, before_texture = _read_textured_ply(Path(before_mesh))
    after_xyz, after_faces, after_texture = _read_textured_ply(Path(after_mesh))
    all_xyz = np.concatenate((before_xyz, after_xyz), axis=0)
    center = (all_xyz.min(axis=0) + all_xyz.max(axis=0)) / 2
    scale = float(np.max(np.ptp(all_xyz, axis=0)))
    scale = max(scale, 1e-6)
    before_positions, before_uvs = _triangles(before_xyz, before_faces, center, scale)
    after_positions, after_uvs = _triangles(after_xyz, after_faces, center, scale)
    anchor_rows = []
    for item in movements or []:
        anchor_rows.append({
            "label": item.get("label", "object"),
            "before": ((np.asarray(item["before_xyz_colmap"], dtype=float) - center) / scale).tolist(),
            "after": ((np.asarray(item["after_xyz_colmap"], dtype=float) - center) / scale).tolist(),
            "distance_m": item.get("estimated_displacement_m"),
        })

    def texture_uri(path: Path) -> str:
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    data = {
        "before": {"positions": before_positions, "uvs": before_uvs, "texture": texture_uri(before_texture)},
        "after": {"positions": after_positions, "uvs": after_uvs, "texture": texture_uri(after_texture)},
        "anchors": anchor_rows,
        "metersPerMapUnit": meters_per_map_unit,
        "worldUp": estimate_colmap_world_up(shared_model) if shared_model else None,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Before and after RGB room maps</title><style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#edf3fa;font:15px system-ui,sans-serif}header{padding:12px 18px;background:#202b38}h1{font-size:21px;margin:0 0 6px}.sub{color:#c5d1de}.views{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}section{min-width:0;background:#1a222c;border:1px solid #3a4757;border-radius:8px;overflow:hidden}h2{margin:8px 12px;font-size:16px}canvas{display:block;width:100%;height:min(63vh,620px);touch-action:none}.legend{padding:8px 14px}.legend span{margin-right:20px}.results{padding:0 18px 15px}.result{display:inline-block;border:1px solid #465568;padding:8px 12px;border-radius:7px;margin:5px}.old{color:#40ed80}.new{color:#ff5260}.arrow{color:#ffdc2d}.error{color:#ff9b9b;padding:8px 14px}@media(max-width:800px){.views{grid-template-columns:1fr}canvas{height:55vh}}
</style><header><h1>Room reconstruction · before and after</h1><div class="sub">RGB textured surfaces. Drag either map to rotate; scroll to zoom; double-click to reset the view. Both maps move together. Green/red markers show old/new object centers on the after map.</div></header>
<div class="views"><section><h2>Before RGB map</h2><canvas id="before" width="900" height="600"></canvas></section><section><h2>After RGB map · old and new centers</h2><canvas id="after" width="900" height="600"></canvas></section></div>
<div class="legend"><span class="old">● Old center</span><span class="new">● New center</span><span class="arrow">━ Movement</span><span>Map axes are normalized for viewing; reported displacement is __SCALE__.</span></div><div class="results" id="results"></div><div id="status" class="error"></div>
<script>
const D=__DATA__,canvases=[document.getElementById('before'),document.getElementById('after')];
let yaw=.65,pitch=.28,zoom=1.65,drag=null,anchorBuffers=[];
const VERTEX=`precision mediump float;attribute vec3 position;attribute vec2 texcoord;uniform mat4 transform;varying vec2 uv;void main(){uv=texcoord;gl_Position=transform*vec4(position,1.0);gl_PointSize=12.0;}`;
const FRAGMENT=`precision mediump float;uniform sampler2D tex;uniform vec4 color;uniform float renderMarker;varying vec2 uv;void main(){gl_FragColor=renderMarker>0.5?color:texture2D(tex,uv);}`;
function fail(message){document.getElementById('status').textContent=message;throw new Error(message)}
function shader(gl,type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){const log=gl.getShaderInfoLog(s);gl.deleteShader(s);fail('WebGL shader compile failed: '+log)}return s}
function makeProgram(gl){const p=gl.createProgram(),vs=shader(gl,gl.VERTEX_SHADER,VERTEX),fs=shader(gl,gl.FRAGMENT_SHADER,FRAGMENT);gl.attachShader(p,vs);gl.attachShader(p,fs);gl.linkProgram(p);gl.deleteShader(vs);gl.deleteShader(fs);if(!gl.getProgramParameter(p,gl.LINK_STATUS))fail('WebGL program link failed: '+gl.getProgramInfoLog(p));return p}
function buffer(gl,data){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);return b}
function setup(canvas,mesh){const gl=canvas.getContext('webgl',{alpha:false,antialias:true,powerPreference:'high-performance'})||canvas.getContext('experimental-webgl');if(!gl)fail('This browser does not provide WebGL. Enable hardware acceleration or use a WebGL-capable browser.');const p=makeProgram(gl);gl.useProgram(p);const pos=buffer(gl,mesh.positions),uv=buffer(gl,mesh.uvs),texture=gl.createTexture(),img=new Image();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([90,90,90,255]));img.onload=()=>{gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);drawAll()};img.onerror=()=>{document.getElementById('status').textContent='Could not decode embedded RGB texture for '+canvas.id+' map.'};img.src=mesh.texture;const locations={position:gl.getAttribLocation(p,'position'),texcoord:gl.getAttribLocation(p,'texcoord'),transform:gl.getUniformLocation(p,'transform'),tex:gl.getUniformLocation(p,'tex'),color:gl.getUniformLocation(p,'color'),renderMarker:gl.getUniformLocation(p,'renderMarker')};return {gl,p,pos,uv,texture,img,locations,count:mesh.positions.length/3,canvas}}
const contexts=[setup(canvases[0],D.before),setup(canvases[1],D.after)];
function normalize(v){const n=Math.hypot(v[0],v[1],v[2])||1;return v.map(x=>x/n)}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function matrix(){const up=normalize(D.worldUp||[0,1,0]),ref=Math.abs(up[2])<.9?[0,0,1]:[1,0,0],right0=normalize(cross(up,ref)),forward0=normalize(cross(right0,up)),cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),right=right0.map((x,i)=>x*cy+forward0[i]*sy),forward=forward0.map((x,i)=>x*cy-right0[i]*sy),screenUp=up.map((x,i)=>x*cp-forward[i]*sp),depth=forward.map((x,i)=>x*cp+up[i]*sp),s=zoom;return new Float32Array([right[0]*s,screenUp[0]*s,depth[0]*s,0,right[1]*s,screenUp[1]*s,depth[1]*s,0,right[2]*s,screenUp[2]*s,depth[2]*s,0,0,0,0,1])}
function resize(c){const dpr=Math.min(window.devicePixelRatio||1,2),w=Math.max(1,Math.round(c.canvas.clientWidth*dpr)),h=Math.max(1,Math.round(c.canvas.clientHeight*dpr));if(c.canvas.width!==w||c.canvas.height!==h){c.canvas.width=w;c.canvas.height=h}c.gl.viewport(0,0,w,h)}
function draw(c,index){const gl=c.gl;resize(c);gl.clearColor(.08,.11,.15,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.useProgram(c.p);gl.uniformMatrix4fv(c.locations.transform,false,matrix());gl.uniform1i(c.locations.tex,0);gl.uniform1f(c.locations.renderMarker,0);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,c.texture);gl.bindBuffer(gl.ARRAY_BUFFER,c.pos);gl.enableVertexAttribArray(c.locations.position);gl.vertexAttribPointer(c.locations.position,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,c.uv);gl.enableVertexAttribArray(c.locations.texcoord);gl.vertexAttribPointer(c.locations.texcoord,2,gl.FLOAT,false,0,0);gl.drawArrays(gl.TRIANGLES,0,c.count);if(index===1&&D.anchors.length){const pts=[];for(const a of D.anchors)pts.push(...a.before,...a.after);if(!anchorBuffers[index])anchorBuffers[index]=buffer(gl,pts);else{gl.bindBuffer(gl.ARRAY_BUFFER,anchorBuffers[index]);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(pts),gl.DYNAMIC_DRAW)}gl.disable(gl.DEPTH_TEST);gl.uniform1f(c.locations.renderMarker,1);gl.bindBuffer(gl.ARRAY_BUFFER,anchorBuffers[index]);gl.enableVertexAttribArray(c.locations.position);gl.vertexAttribPointer(c.locations.position,3,gl.FLOAT,false,0,0);gl.disableVertexAttribArray(c.locations.texcoord);gl.vertexAttrib2f(c.locations.texcoord,0,0);gl.uniform4f(c.locations.color,1,.84,.08,1);gl.drawArrays(gl.LINES,0,pts.length/3);for(let i=0;i<D.anchors.length;i++){gl.uniform4f(c.locations.color,0.15,.95,.42,1);gl.drawArrays(gl.POINTS,i*2,1);gl.uniform4f(c.locations.color,1,.25,.29,1);gl.drawArrays(gl.POINTS,i*2+1,1)}gl.enable(gl.DEPTH_TEST)}}
function drawAll(){contexts.forEach(draw)}
canvases.forEach(c=>{c.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)});c.addEventListener('pointerup',()=>drag=null);c.addEventListener('pointercancel',()=>drag=null);c.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.2,Math.min(1.2,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];drawAll()});c.addEventListener('wheel',e=>{zoom=Math.max(.6,Math.min(5,zoom*(e.deltaY<0?1.08:.92)));drawAll();e.preventDefault()},{passive:false});c.addEventListener('dblclick',()=>{yaw=.65;pitch=.28;zoom=1.65;drawAll()})});window.addEventListener('resize',drawAll);if(window.ResizeObserver){const ro=new ResizeObserver(drawAll);canvases.forEach(c=>ro.observe(c))}document.addEventListener('visibilitychange',()=>{if(!document.hidden)drawAll()});drawAll();
document.getElementById('results').innerHTML=D.anchors.length?D.anchors.map(a=>'<div class="result"><b>'+String(a.label).replace(/[&<>"']/g,'')+'</b>: old center → new center'+(a.distance_m==null?'':' · '+Number(a.distance_m).toFixed(2)+' m')+'</div>').join(''):'<p>No automatic movement match passed the identity/geometry confidence checks.</p>';
</script>"""
    scale_label = "meters from this run's provisional TV calibration" if meters_per_map_unit else "unscaled COLMAP units"
    page = page.replace("__DATA__", payload).replace("__SCALE__", scale_label)
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output
