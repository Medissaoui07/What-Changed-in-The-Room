# Local two-video room-change UI

From `navigation_humanoid/`, use a project-local Python environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ui,models,depth]'
python -m streamlit run src/what_changed_room/ui_app.py --server.maxUploadSize 2048
```

The UI needs COLMAP on `PATH`. Building the **paired RGB room maps** also
requires the OpenMVS executables `InterfaceCOLMAP`, `DensifyPointCloud`,
`ReconstructMesh`, `RefineMesh`, `TextureMesh`, and `Viewer`. See the main
[README setup section](../README.md#requirements) for native-tool installation
links and a command that checks the default `/usr/local/bin` location. Set the
OpenMVS tools folder in the UI if they are installed elsewhere.

Internet access is needed on the first run to download the default YOLO,
DINOv2, and Depth Anything checkpoints. Model selection and model-folder
settings are intentionally hidden in the UI; it uses the defaults listed
below. CPU mode is supported; GPU is optional.

Upload the before and after videos. The app preprocesses both, reconstructs a
sparse COLMAP reference, localizes after frames, runs YOLO segmentation for all
available classes, and triangulates provisional object centers when at least
four registered views support them. There is no manually entered class list.

## Results screen

- **Detected changes** lists moved, observed, possible addition/removal, and
  uncertain candidates. DINOv2 compares masked appearance tracks automatically.
  It uses the multiview anchor to form a capture-local track and compares
  several masked views per capture. Current similarity cutoffs are a 0.60
  median and a 0.50 lower quartile. These are demo heuristics, not validated
  identity guarantees; different instances of the same class can still be
  confused, and uncertain candidates are not forced into matches.
- **Paired RGB evidence** shows mask-preview frames from both captures.
  Movement candidates can project old and new 3D centers into registered after
  photos. Red is the old center; green is the new center.
- **Paired RGB room maps** are generated automatically when the Results tab is
  rendered. OpenMVS builds side-by-side textured reconstructions with old/new
  anchors and movement on the after map. This CPU stage may take several
  minutes. Drag to rotate, scroll to zoom, and double-click to reset the view.
  Results are display-only; map navigation is gesture-based.
- **Possible removals** remain candidates until the old full object footprint
  is visible and empty from multiple registered after views. Detector misses,
  occlusion, and missing sparse points never prove removal. Projected old-center
  evidence is shown automatically for up to three after views per candidate;
  the UI does not manually confirm or save a removal decision. The Results tab
  has no action, save, or review buttons; processing starts in the Upload tab.

## Metric scale

Every COLMAP reconstruction has its own arbitrary scale. The app therefore
does not reuse the previously measured TV factor. It looks for the known TV in
at least two registered before frames, calibrates Depth Anything V2 depth to
that run's COLMAP camera poses, and estimates a fresh meters-per-map-unit
factor assuming fixed reference TV dimensions of 0.90 × 0.55 m; these values
are not editable in the UI. If the TV in the scene has different dimensions,
the metric estimate may be wrong. The app accepts an estimate only when
repeated views agree within the calibration rule; otherwise distances stay in COLMAP units. The
reported scale and its source are saved in each run's
`metric_scale_calibration.json`. This is a provisional scale estimate, not a
survey-grade measurement.

The defaults use the downloadable model IDs `yolo11n-seg.pt`,
`facebook/dinov2-small`, and `depth-anything/Depth-Anything-V2-Small-hf`. DINOv2
matching falls back to uncertain results if its model cannot load. TV scale
calibration also remains explicitly uncalibrated when the TV is absent, not
detected in enough registered views, or the depth fit is rejected. Uploads and
results stay local under `output/ui_runs/<capture_id>/`; model weights are
cached by their respective Python libraries.
