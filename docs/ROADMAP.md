# Development Roadmap

## Challenge and project goal

Given two phone videos of the same small indoor area, produce a reviewable comparison that:

1. puts evidence from both recordings in one shared 3D coordinate frame;
2. finds and visualizes likely moved, added, or removed objects/regions;
3. describes candidates with semantic labels or short natural language;
4. distinguishes real changes from viewpoint, occlusion, and missing evidence.

The submission should show an end-to-end result on a deliberately staged room, not just successful intermediate geometry. Every claim must link to frames and 3D evidence. The system must abstain when evidence is weak.

## Revised technical approach

The first plan over-weighted registering two independently reconstructed sparse clouds. That is a hard subproblem, and the current recordings demonstrate the limitation: only 9 cross-model feature matches were found and just 1 was geometrically consistent. The registration implementation is useful as a fallback and synthetic test target, but the current real pair is not ready for change analysis.

Use the **before COLMAP sparse model as the shared map**. Localize selected after-video frames into that existing model using COLMAP image registration / 2D-to-3D matching against stable room structure. This directly puts the after camera poses in the before coordinate frame and avoids estimating a transform between two weak independent maps. Keep changed objects out of the alignment evidence where possible. If too few after frames register, stop and report that capture/reconstruction needs improvement.

Use sparse geometry for camera poses, stable anchors, and 3D point evidence; use the original RGB frames for object-level evidence. Detect/segment a small set of visible objects in selected frames, compare corresponding observations across recordings, and associate mask pixels with COLMAP observations/3D points when available. Visualize the result over the shared sparse map and source frames. A changed region with no reliable 3D support remains an image-space candidate; do not invent its depth or claim a full object surface.

Semantics should be practical and bounded: start with a lightweight local object detector for common room objects, with manual labels available for the staged demo. A vision-language model may improve descriptions later, but it is not needed to establish whether geometry and image evidence support a candidate. Model labels never establish absence or override failed localization.

### What “shared 3D representation” means for this project

For the sparse baseline, it means one COLMAP map and both sets of camera poses expressed in that map's coordinates, plus changed-region anchors where image features can be linked to 3D points. It does not mean a complete mesh or occupied volume. If the challenge evaluator expects object surfaces rather than sparse anchors, the sparse-only representation is a limitation and must be stated; do not present a feature cloud as a complete room scan.

## Status and immediate next action

- Phases 0–2: implemented (setup, video preprocessing, point-cloud interface/import).
- Phase 3A: shared-map localization implemented. On `output/new_capture_run`, 16 before cameras, 4 after cameras, and 770 sparse points are present in the best preview.
- Phase 4: observation annotation/tracking and optional sparse-point linking are implemented; those sparse points may be background.
- Phase 5: conservative comparison/reporting and manual evidence review are implemented. The current laptop pair remains uncertain; it does not meet the moved/added/removed demo acceptance criteria.
- **Current: GPU model integration and deadline demo.** YOLO11n-seg has a `wcr detect-objects` adapter. Depth Anything V2 Small calibration passed on three selected frames; `wcr backproject-objects` created provisional geometry for 21 YOLO masks and 47,360 points in `../runs/object_geometry/`. DINOv2-S ranked the before/after masked crops; top pairs look plausible but are still suggestions. The review page records identity and geometry decisions, and `wcr compare-reviewed-matches` produces conservative movement candidates from explicit human review. Keep SAM2/LightGlue/open-vocabulary models as targeted fallbacks.
- **Chair-test update:** depth calibration alone does not cover the chair well in both captures. An optional `--triangulate-label chair` path now intersects repeated mask-center rays using registered COLMAP poses, rejects outlier views, and reports a provisional 3D center with residual/parallax diagnostics. On `runs/chair_test`, it found 9 consistent before views and 15 after views; the reviewed centers are 5.77 COLMAP units apart. This is evidence for a likely move, not a metric displacement or a confirmed object surface. The chair review and output are in `../runs/chair_test/matches/chair_retest_review.json` and `chair_retest_candidates.html`.

The current models contain 179 points / 4 registered before images and 469 points / 7 registered after images. Their independent-model matcher finds 9 pairs with 1 inlier, so no transform is returned. That is a valid fail-safe outcome, not a detected room change.

## Phase 3A — Shared-map localization (next)

- Preserve before model, its database, and source frames as the reference map.
- Add selected after keyframes to a COLMAP database and attempt image-to-model registration against stable mapped structure.
- Record which after images localized, their reprojection/inlier diagnostics, and their camera poses in before-map coordinates.
- Require adequate spatial coverage across both recordings; fail with a recapture/reconstruction explanation when registration is weak.
- Keep the independent sparse-cloud similarity registration as a fallback experiment, not the main route.
- Add a simple visualization of the reference sparse map, before cameras, and localized after cameras.

**Acceptance:** several after views from different positions localize consistently to the before map; known stable landmarks overlap in the shared view; weak localization is rejected. Never call the maps shared just because their coordinates look similar.

The first runnable command is `wcr localize-colmap`; it writes `shared_model/` and `shared_map.html`. This is the first visible geometry result. It validates camera localization only; object changes and semantic labels are later milestones.

## Phase 4 — Visual candidate observations (implemented; detector proposals available)

- Accept manually annotated or detector-exported boxes/polygons for selected RGB frames through `wcr observations` and a small `ObjectDetector` protocol. `wcr detect-objects` now provides an optional Ultralytics YOLO segmentation adapter; model weights remain user-managed.
- Track within a capture using supplied track IDs or provisional adjacent-frame box overlap; IDs are capture-namespaced and do not assert before/after identity.
- Link a region to COLMAP point IDs and coordinates only when the registered image has observed mapped keypoints inside that region.
- Preserve unregistered frames and regions without mapped keypoints as `image_space_only`, with no invented depth or anchor.
- Do not compare before/after tracks or emit change labels in Phase 4; that is Phase 5.
- Missing sparse points alone are never removal evidence.

**Acceptance:** annotations retain frame/box/label evidence and can link existing sparse anchors, or explicitly remain image-only. No Phase 4 output turns missing geometry into an absence/removal claim. Cross-view association quality remains for Phase 5.

**Current YOLO run:** all 96 source frames produced 190 detections (103 before,
87 after); 112 provisional/single-observation tracks remain `insufficient_observation`.
The model confuses some room regions with unrelated COCO classes, and its masks
do not establish instance identity. Only 4 after frames are localized into the
shared model, so downstream 3D claims remain limited. Reports and previews are
in the parent workspace's `runs/segment/` directory; `predict_all/` contains
new previews for both captures. PyTorch selected CPU in
this execution because CUDA was unavailable to the active `.myenv`.

## Phase 5 — Candidate association and change reasoning (implemented; demo acceptance pending)

- `wcr compare-observations` proposes a one-to-one same-label association only when each capture has exactly one track of that label. This is a candidate link, not an identity claim; ambiguous/multiple same-class tracks stay unmatched and uncertain.
- Include overlap among linked sparse point IDs as map-context evidence only. These points can lie on background, so they never establish object identity or object displacement.
- Allow a reviewer to confirm identity and provide object-specific anchors selected manually in the shared map. Call `moved` only when reviewed identity and both anchors exist and their displacement exceeds an explicit threshold in shared-map units.
- Allow a reviewer to mark an unmatched track's opposite region `fully_visible`; only then report `candidate_added`/`candidate_removed`. Occluded, partial, unobserved, or unknown regions remain `insufficient_observation`.
- Keep candidate type, association status, semantic label, geometry evidence, visibility, and uncertainty reasons as separate report fields.
- Never infer removal from missing sparse points.

**Acceptance:** automated reasoning is conservative and reviewable. The current real pair does not yet satisfy the demonstration criterion of one clear moved/added object plus one unchanged/occluded example; object-level anchors and visibility must be manually established or the capture should be improved.

## Phase 6 — Report and visualization

- Write a JSON report with candidate type, label, shared-frame location/anchor, before/after evidence frames, localization quality, visibility, and uncertainty reasons.
- Visualize the shared sparse map and registered camera paths; show candidate anchors/regions and linked before/after frames.
- Include a compact natural-language summary derived from the structured candidate record. A VLM may phrase or suggest labels from supplied crops, but may not supply coordinates or assert physical absence.
- Clearly label map geometry as sparse and incomplete.

**Acceptance:** a reviewer can follow each claim from summary to candidate, 3D view, and source frames. Failed/uncertain candidates remain visible as uncertain rather than disappearing from the report.

## Phase 7 — GPU perception and demo evaluation

- Capture a short, controlled pair with stable room background, enough overlapping views, and 1–3 obvious changes.
- Record ground truth before processing and include an unchanged or partially occluded object.
- Measure shared-map localization success, candidate precision/recall on the small staged set, abstentions, runtime, and failure cases.
- Run YOLO11n-seg on selected RGB frames and preserve reviewed masks/track IDs.
- `wcr estimate-depth` has produced accepted camera-Z maps for selected before/after frames. `wcr backproject-objects` now samples masks only from frames with accepted calibration and transforms those pixels through the COLMAP camera model and pose into shared-map coordinates. YOLO masks remain provisional pending review; this does not classify changes or prove removal.
- Use `wcr rank-object-matches` to run DINOv2 ViT-S/14 crop embeddings and rank cross-capture object candidates. Require reviewer confirmation for physical-instance identity.
- Review candidate crops and mask-derived geometry in `dino_match_review.html`; export decisions and run `wcr compare-reviewed-matches`. Do not infer additions or removals from missing observations.
- Keep manual correction and image-space evidence available when a model or calibration fails. Record model versions, checkpoint sizes, license, GPU name/VRAM, runtime, and peak memory.
- Add SAM2, LightGlue, OpenCLIP, or Grounding DINO only after a visible failure establishes a need for that specific capability.

**Acceptance:** show a complete video-to-report run and state exactly which outputs are automated, manually labeled, or uncertain. Do not claim general object-change accuracy from a single room pair.

## One-day GPU plan

The user reports NVIDIA drivers are available. Use the existing `.myenv` and
verify that its PyTorch build can access CUDA; driver presence by itself does
not establish PyTorch/CUDA compatibility. Do not rebuild COLMAP with dense CUDA:
the current sparse reference map remains the geometric backbone.

Required default models are deliberately small and serve distinct tasks:

- **YOLO11n-seg** (~6.2 MB checkpoint): common-class detections and instance
  masks. The project must accept Ultralytics AGPL-3.0 terms or obtain its
  Enterprise license. Correct masks manually when the staged object is missed.
- **Depth Anything V2 Small** (~99.2 MB): dense relative depth for selected
  frames. Calibrate each frame against visible COLMAP points and reject failed
  fits; output coordinates remain arbitrary COLMAP units, not meters.
- **DINOv2 ViT-S/14** (~88.2 MB safetensors): rank cross-capture crop appearance
  matches; a reviewer confirms the same physical instance.

Approximate combined checkpoint downloads are 194 MB, excluding packages and
existing Torch/CUDA. Run only on 4–8 selected frames and record actual inference
time/VRAM on the user's GPU. SAM2 Tiny is a mask-refinement fallback; LightGlue
is a feature-matching fallback; OpenCLIP/Grounding DINO are open-vocabulary
fallbacks. Install one only when its stated failure case occurs. Full setup,
depth calibration gates, visual outputs, and work allocation are in the
[model plan](MODEL_RESEARCH_AND_1DAY_PLAN.md) and [workflow schema](WORKFLOW.md).

**Immediate action:** enable network access to Hugging Face and CUDA in `.myenv`,
then run `wcr estimate-depth` on the two or three registered frames described
in the README. The `[depth]` dependencies are installed, but the checkpoint
could not be fetched because `huggingface.co` did not resolve, and PyTorch
currently reports CUDA unavailable. Inspect `depth_calibration.json` and
previews before any object back-projection. Do not back-project masks if
calibration or camera coverage fails; preserve image-space mask evidence and
uncertainty.

## Deferred scope

- Dense COLMAP, mesh reconstruction, and CUDA builds: deferred because the attempted build blocked progress and the current demonstrator can use sparse camera/feature anchors.
- SAM2 mask propagation, LightGlue, open-vocabulary detection, agent workflows, and a web UI: optional extensions after the selected-frame GPU pipeline works end to end.
- Do not spend the deadline adding model size before the capture pair can localize and produce a traceable candidate.
