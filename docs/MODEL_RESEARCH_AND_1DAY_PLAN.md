# Model choices and one-day GPU plan

## Decision

Use three pretrained models for three separate jobs, and keep COLMAP/OpenCV for
geometry and camera alignment:

1. **YOLO11n-seg** detects common room objects and returns instance masks.
2. **Depth Anything V2 Small** predicts a dense relative-depth image for each
   selected frame; calibrate it against visible COLMAP points before using it
   to make any shared-map 3D estimate.
3. **DINOv2 ViT-S/14** compares object-crop appearance to rank possible
   before/after matches.

This is a GPU-capable, offline pipeline, not a real-time robot stack. The first
goal is a small controlled demo with 1–3 staged changes. Run models only on
selected keyframes and detected crops, not every decoded video frame.

## Model selection by task

| Job | Selected model | Download size | Why this choice | What it cannot establish |
| --- | --- | --- | --- | --- |
| Common-object masks | YOLO11n-seg | About 6.2 MB checkpoint | Smallest YOLO11 segmentation variant; combines class labels, boxes, and instance masks in one pass. Start with COCO furniture classes. | Classes are closed-set; a mask is only a proposal. It cannot identify the same physical instance across recordings or prove an object is absent. Ultralytics licensing is AGPL-3.0 or Enterprise; check project/company terms. |
| Dense monocular depth | Depth Anything V2 Small (ViT-S) | 99.2 MB checkpoint | Lightest official DA-V2 encoder; suitable for offline inference on a few RGB frames. The checkpoint is Apache-2.0. | Output is relative depth, not meters and not COLMAP-aligned. Calibration can fail when sparse anchors are few, clustered, or have large residuals. |
| Crop appearance / object matching | DINOv2 ViT-S/14 | About 88.2 MB safetensors, or 84 MB-class backbone checkpoint | Compact generic visual features; use cosine similarity to rank candidate object crops across views. Apache-2.0 standard weights. | Similar objects can look alike; viewpoint, lighting, and occlusion change embeddings. A high score is not proof of instance identity. |

Approximate combined checkpoint downloads: **about 194 MB** for these selected
files, excluding Python packages, Torch/CUDA already installed, cache copies,
and any exported models. Sizes depend on checkpoint format and mirror. DINOv2's
official ViT-S is 21M parameters; Depth Anything V2 Small is 24.8M parameters.
YOLO11n-seg checkpoint is listed at 6.2 MB. These are model files, not peak GPU
memory requirements.

### Alternatives: install only if the default misses a demonstrated need

| Alternative | Checkpoint / burden | Add it when | Do not use it for |
| --- | --- | --- | --- |
| SAM 2.1 Hiera Tiny | 156 MB PyTorch checkpoint (Transformers safetensors variant about the same); needs compatible PyTorch/TorchVision and may compile a CUDA extension. Apache-2.0. | YOLO finds the object but its boundary is poor, or a prompted mask needs propagation through adjacent frames **within one capture**. | Semantic class detection or deciding that two captures show the same object. It overlaps with YOLO mask work, so it is not in the first install batch. |
| LightGlue + SIFT | Additional matcher code/weights; SIFT extractor avoids SuperPoint's separate restrictive terms. | OpenCV/COLMAP feature matches fail on a particular pair and stronger local correspondences could rescue image alignment. | Object identity: matches may come from static background. |
| OpenCLIP ViT-B/32 | Usually a larger text-image checkpoint; exact size and license depend on the chosen pretrained weights. | Natural-language zero-shot class ranking is required and YOLO's known classes do not cover the object. | Metric geometry or physical-instance identity. Prefer DINOv2 for the first crop-similarity experiment. |
| Grounding DINO | Larger dependency/checkpoint and slower setup than YOLO nano. | Text-prompted open-vocabulary boxes are specifically needed for an object YOLO cannot label. | Mask refinement, tracking, depth, or cross-recording identity. Not justified for common furniture in this deadline. |
| Depth Anything V2 Base/Large | 97.5M / 335.3M parameters; larger downloads and higher memory use. | Small model visibly fails on relevant boundaries and GPU memory/time allow a measured comparison. | Fixing failed COLMAP localization or creating metric scale by itself. |

Do **not** install SAM2, LightGlue, OpenCLIP, and Grounding DINO together just
because they are available. Each is an escalation for a specific observed
failure. YOLO and DINOv2 do different things: YOLO proposes labeled instances;
DINOv2 provides appearance similarity between selected crops.

## Pipeline and evidence rules

1. **Preprocess** both videos and keep source frame names/timestamps.
2. **Build the shared sparse map:** reconstruct the before capture with COLMAP,
   then localize selected after frames into that reference model. This is the
   camera-coordinate backbone. Do not rebuild COLMAP with dense CUDA just to
   use these models.
3. **Gate on localization and overlap:** select pairs where both RGB views show
   the same stable room area. Failed or weak localization remains a failure.
4. **Detect and segment:** run YOLO11n-seg on selected frames. Save class,
   confidence, box, polygon/mask, frame, and capture-scoped provisional track.
   Review the masks; manually correct important staged objects.
5. **Estimate depth:** run DA-V2 Small on those frames. Project COLMAP 3D points
   into each registered image, sample predicted depth at valid visible points,
   and fit a robust scale-and-offset mapping from model depth to that frame's
   camera-space COLMAP depth. Keep a holdout set and report anchor count,
   image coverage, inlier ratio, and error. Reject calibration if anchors are
   too few/clustered or holdout error is high. The resulting values are in
   arbitrary COLMAP map units, not meters.
6. **Make object regions in 3D only after calibration:** combine the instance
   mask with calibrated depth and camera intrinsics/pose to back-project pixels
   into the shared coordinate frame. Filter uncertain boundaries and report
   depth confidence/residual. If any prerequisite fails, retain image-space
   evidence and no fabricated 3D location.
7. **Associate object candidates:** compute DINOv2 embeddings for masked crops;
   rank before/after candidates using appearance plus label, visibility, and
   their estimated shared-map geometry. Keep association uncertain for similar
   instances or poor views; a human confirms identity for a one-day demo.
8. **Reason about change:** call “moved” only for the same reviewed object with
   supported 3D positions in both captures. A new/removed claim also needs
   reviewed visibility of the expected region. A missing sparse point or
   detector miss never proves removal.
9. **Visualize:** show paired RGB frames, masks, depth heatmaps, calibration
   anchors/residuals, candidate 3D points in the sparse map, and uncertainty.

Depth calibration is a research/engineering bridge, not a guaranteed property
of monocular depth. If the before-map anchors do not constrain the prediction
well, show a relative-depth visualization only and keep the object image-space.
Do not describe a dense back-projection as a complete or metrically accurate
room scan.

## One-day plan (about 8 focused hours)

| Time | Work | Completion check |
| --- | --- | --- |
| 0:00–0:30 | Confirm the active `.myenv` sees CUDA; record GPU name, VRAM, PyTorch/CUDA versions, and disk space. Confirm `colmap --version`. | `torch.cuda.is_available()` is true and a short CUDA tensor operation succeeds. |
| 0:30–1:15 | Install the YOLO package in `.myenv`; run `yolo11n-seg.pt` on 2–4 representative frames. Fetch DINOv2-S and DA-V2-S checkpoints. | All three load on CUDA; save a timing and memory log. Stop on dependency conflict and preserve the existing environment. |
| 1:15–2:00 | Run the existing shared-map localization and select 4–8 well-overlapped frames total. | Several useful before/after views are in the same map and show the target room region. |
| 2:00–3:00 | Generate YOLO masks on those frames; visually review boxes/masks and correct staged-object identity/track IDs. | One known changed object and one stable or occluded control have reviewed masks. |
| 3:00–4:00 | Generate DA-V2-S depth maps; project sparse map points into frames and test robust scale/offset calibration with holdout anchors. | Per-frame diagnostic says calibrated or rejected; no unsupported depth is exported. |
| 4:00–5:00 | Back-project only supported mask pixels; compute object region/centroid and uncertainty in COLMAP coordinates. | Before and after object locations are linked to source frame and calibration diagnostics. |
| 5:00–5:45 | Generate DINOv2 crop similarities and shortlist cross-capture associations. | Candidate pair list is human-reviewed; no similarity score is presented as proof. |
| 5:45–7:00 | Produce paired-frame and shared-map visualization; report moved/added/removed or abstain. | One reviewer can trace each conclusion to RGB, mask, depth fit, pose, and object anchor. |
| 7:00–8:00 | Run one unchanged/occluded control and one poor-view control; write limitations and demo commands. | Controls do not produce confident changes; poor evidence triggers abstention. |

These are work allocations, not guaranteed runtimes. Model setup/download time
depends on network and installed CUDA/PyTorch compatibility; GPU inference
depends on GPU, image resolution, precision, and VRAM. Time a handful of frames
before choosing a larger variant. Use FP16/autocast only after checking output
quality and GPU support.

## Start here

The YOLO adapter has been added and run on the 41 before and 55 after frames.
The parent-level `runs/segment/` now contains the previous before overlays,
new overlays for both captures in `predict_all/`, `annotations_yolo_all.json`,
`observations_yolo_all.json`, and `change_candidates_yolo_all.json`. The YOLO
run found 190 detections; all 112 resulting candidates abstain as
`insufficient_observation`. This is expected: masks are not instance identity,
and only four after frames localize into the sparse shared map.

Next:

1. Inspect original before overlays in `runs/segment/predict/` and the new
   all-frame overlays in `runs/segment/predict_all/`; manually select the correct object masks
   on registered overlapping frames. The model produces some false class
   labels, so do not treat its full-room detections as ground truth.
2. Verify CUDA access in `.myenv`; the last inference selected CPU because
   `torch.cuda.is_available()` was false in that environment.
3. Use the local Depth Anything V2 Small weights in
   `~/models/depth-anything-v2` with `wcr estimate-depth`. Ensure the directory
   also contains `config.json` and `preprocessor_config.json`; the README has
   commands to fetch those metadata files. The command fits relative depth
   against COLMAP camera-space anchor depths with holdout error and
   spatial-coverage checks. If fit fails, keep depth as visualization only and
   object evidence image-space.
4. After depth and localization evidence are sound, use
   `wcr rank-object-matches` to rank cross-capture DINOv2-S crop matches for
   human review. Similarity remains a proposal, not instance identity.

## Official sources and current model details

- [Ultralytics YOLO11 task table, COCO classes, performance, and license](https://docs.ultralytics.com/models/yolo11); [YOLO11n-seg checkpoint listing](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11n-seg.pt).
- [Depth Anything V2 official repository, model sizes, and license](https://github.com/DepthAnything/Depth-Anything-V2); [official Small checkpoint card and 99.2 MB file](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/tree/main).
- [DINOv2 official model table and 21M-parameter ViT-S/14](https://github.com/facebookresearch/dinov2); [Meta model card and 88.2 MB safetensors file](https://huggingface.co/facebook/dinov2-small/tree/main).
- [SAM 2 official repo, installation, checkpoint, and benchmark](https://github.com/facebookresearch/sam2); [SAM2.1 Tiny 156 MB checkpoint](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/main).
- [LightGlue official repo, benchmarks, and licenses](https://github.com/cvg/LightGlue).
- [OpenCLIP repo](https://github.com/mlfoundations/open_clip) and [checkpoint-specific metadata](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md).
- [Grounding DINO official repository](https://github.com/IDEA-Research/GroundingDINO).
- [OpenCV image geometry documentation](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html).
