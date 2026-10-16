# Architecture

## 1. Product goal

Compare two handheld videos of the same room and produce an evidence-backed list of **possible spatial changes**. Each result should show where it may have happened, which frames support it, how well the scenes aligned, and what remains uncertain. The system should ask for a better capture when the recordings do not support a useful conclusion.

The practical baseline uses **COLMAP sparse geometry plus RGB image evidence**. A sparse model gives camera poses and feature anchors, not complete room surfaces. The deliverable is an evidence-linked set of object/region change candidates in a shared coordinate frame, with uncertainty shown explicitly. It must not claim complete surface or occupancy reconstruction.

## 2. End-to-end flow

```text
Before/after video
  -> sampled frames and timestamp manifest
  -> COLMAP sparse map from the before capture
  -> localize after keyframes into that same map
  -> object observations from RGB frames linked to sparse 3D anchors where possible
  -> conservative change association and visibility checks
  -> evidence-linked report, shared-map visualization, and uncertainty
```

Each stage has a replaceable interface. The core pipeline remains usable without neural weights or paid services.

## 3. Inputs and reconstruction

The pipeline accepts the existing imported PLY/PCD point-cloud format and a COLMAP sparse model. Keep the original COLMAP model and its registered images, camera poses, and image names; exporting to PLY alone loses camera and observation metadata needed by later stages.

COLMAP sparse points are triangulated image features. Their density depends on texture, viewpoint, and feature matching; they do not sample every visible surface. Use them as camera-pose/registration anchors and linkable 3D evidence, not as a dense room scan. The primary path is one reference map plus after-frame localization into it, rather than two independently reconstructed maps that then need a fragile cross-model alignment. Keep independent-map similarity registration as a fallback. Do not build or require CUDA dense MVS for the current baseline.

Record, when available:

- source video and frame-manifest identifiers;
- registered image names and camera poses;
- sparse points and colors;
- reconstruction/model ID and registered-image count;
- whether the scale is metric, externally known, or unknown;
- reconstruction warnings and provenance.

If independent reconstructions are used, they must not be assumed to share axes, origin, orientation, or scale. COLMAP point IDs are local to one model; equal IDs across models do not indicate correspondence. A PLY export drops image observations/tracks. Preserve native models, databases, registered images, and frames. In the preferred shared-map path, both sets of camera poses are expressed in the reference model's coordinates after successful image-to-model localization.

## 4. Registration

Estimate a transform from the after-scene coordinates into the before-scene coordinates. With two monocular reconstructions, use a **similarity transform** (scale, rotation, translation) unless a known measurement or other metric source establishes scale. Do not silently interpret arbitrary reconstruction units as meters.

Use stable room structure to localize after keyframes into the reference COLMAP model (2D-to-3D matching/PnP through COLMAP image registration). The existing independent-model implementation reads each model's tracked SIFT descriptors, forms cross-model feature matches, and estimates a similarity transform with RANSAC; it is a fallback. On the current real pair it finds 9 candidates but only 1 inlier, so that route is not validated. PLY-only geometric alignment is even less informative because it loses image tracks. Evaluate inliers, spatial distribution, reprojection/residual error, and coverage. Poor or ambiguous localization must stop confident downstream classification.

Document the transform direction and coordinate frames in the API and report. Keep registration quality separate from any later semantic confidence.

## 5. Observation and uncertainty

Sparse PLY geometry alone cannot establish that an empty-looking region was observed. Camera poses and source frames may provide a **coverage proxy**, but sparse feature visibility is not equivalent to full surface visibility. Distinguish:

- supported observation;
- possible observation based on camera/frame evidence;
- occluded or out of view;
- insufficient observation;
- uncertain reconstruction or alignment.

If camera poses or image links are unavailable, mark visibility as unknown. A point missing from one sparse model is never, by itself, proof that an object was removed.

## 6. Change hypotheses and object association

Detect or manually annotate object regions in selected RGB frames, associate observations across views, and link pixels to COLMAP 3D tracks/anchors when possible. Compare before/after observations only after after-frame localization into the shared map. Report an image-space candidate when there is no reliable 3D anchor; never fabricate depth. Sparse clouds can support camera geometry and point anchors, but cannot by themselves prove an object surface was occupied or empty.

Associate before/after candidates only when the evidence supports correspondence. A moved object needs evidence for both its old and new locations and a plausible identity match. If movement cannot be distinguished from unrelated addition/removal, say so. Never infer missing surfaces from sparse sampling.

## 7. Optional visual models

Visual models add evidence; they do not replace shared-map localization or observation checks. For the selected-frame GPU demo, use a small local model for each distinct role:

- **YOLO11n-seg:** common-class proposals and instance masks. Review its labels and masks; its AGPL-3.0/Enterprise licensing needs to fit the project. A mask does not prove cross-capture identity or absence.
- **Depth Anything V2 Small:** dense relative-depth maps. Fit each selected frame to camera-space depth from projected COLMAP anchors and validate on held-out anchors before using mask pixels as 3D evidence. The output is not meters. If calibration fails, preserve image-space evidence.
- **DINOv2 ViT-S/14:** crop-appearance similarity for ranking candidate object matches. A similarity score does not prove the same physical instance.
- **SAM2, LightGlue, OpenCLIP, Grounding DINO:** targeted fallbacks for mask refinement, difficult feature matching, or open-vocabulary labels, respectively. Do not add them without a demonstrated need.
- **VLM:** optional descriptions or bounded labels for selected before/after crops. Ask for structured candidate labels tied to supplied frame IDs; do not ask it to invent coordinates or determine physical absence from an unseen area.

Run visual models only on selected keyframes/candidate crops. No model weights, external API, or network access are required for the geometry baseline. Never download large weights without explicit approval. Validate proposed labels against geometry and multi-view image evidence; use an uncertain status when evidence conflicts.

## 8. Result schema

Each candidate should include:

- stable candidate ID and one of `moved`, `added`, `removed`, `possible_change`, `insufficient_observation`;
- optional semantic label and the source/model that proposed it;
- before/after 3D locations in declared frames, when available;
- scale status (`metric`, `estimated`, or `unknown`);
- registration diagnostics and observation status;
- evidence point/region IDs and before/after frame IDs with timestamps;
- uncertainty reasons and confidence only when calibrated by evaluation.

Use null for unavailable values. Do not label raw model certainty as calibrated confidence.

## 9. Visualization and user interaction

Show sparse points and registered camera paths, transform/registration diagnostics, candidate locations, and linked source frames. Clearly label the view as sparse evidence rather than a complete surface model. The JSON and visualization must agree.

When a candidate is uncertain because of poor coverage, report what view is missing where feasible, so a user can record a short follow-up video. Keep this guidance advisory until it is validated.

## 10. Performance and dependencies

Keep the baseline offline and CPU-capable. COLMAP sparse mapping/localization does not require the failed CUDA dense-build workflow. Keep NumPy/SciPy and existing import contracts central; a small detector is optional, while manual labels are a practical fallback for the staged demo. Build the CLI, evidence report, and visualization before a UI or large neural models.
