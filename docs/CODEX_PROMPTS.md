# Implementation Prompts

Use these as scoped work items. Read `AGENTS.md`, `README.md`, `CODEX.md`, and the relevant docs before implementation. Follow `ROADMAP.md`. The challenge deliverable is an evidence-linked comparison in shared 3D, not a particular model or reconstruction tool. COLMAP is replaceable.

## Prompt 1 — Repository inspection

Inspect repository state and environment. Report dependencies, available geometry tools, risks, and a practical implementation order. Implement only Phase 0 of `ROADMAP.md`.

## Prompt 2 — Video preprocessing

Implement Phase 1 only: validate two videos, sample frames with timestamps, save a manifest, provide useful errors, and add tests. Keep dependencies minimal.

## Prompt 3 — Geometry backend interface and point-cloud import

Implement Phase 2 only: define a replaceable mapping/localization result/backend contract and import PLY/PCD. Preserve colors and coordinate/scale provenance where available. Add small fixtures and tests; do not require a specific mapper or large model dependency.

## Prompt 4 — Shared-map localization (next)

Implement Phase 3A only. Prefer building a reference map from the before capture and localizing after keyframes into it, so camera poses share coordinates. COLMAP may provide the first backend, but keep it replaceable. Report registered frames, spatial coverage, inlier/reprojection evidence, and clear failure reasons. Preserve independent-map similarity registration as a fallback. Never claim shared coordinates without passing measured checks. Visualize the map and both camera paths.

## Prompt 5 — Object observations (implemented)

Implement Phase 4 only. Add interfaces for object detections or manual annotations on selected RGB frames. Track observations within each capture and link them to 3D points/anchors where possible. Mark unsupported detections as image-space evidence without invented depth. Missing sparse points must never prove removal. Keep model choice optional and lightweight.

## Prompt 6 — Cross-capture change reasoning (implemented; real-pair evaluation pending)

Implement Phase 5 only. Associate observations using shared-map camera geometry and image evidence. Distinguish movement from addition/removal only with adequate evidence. Preserve occlusion, unknown visibility, and ambiguous cases explicitly. Add tests for unchanged, moved, added, removed, viewpoint-only, and partially observed scenes.

## Prompt 7 — Report and visualization

Implement Phase 6 only. Produce JSON and a readable shared-map visualization with camera paths, candidate anchors/regions, linked before/after frames, localization status, and uncertainty. Clearly label sparse geometry as incomplete and image-only candidates as lacking a 3D anchor. Ensure report and visualization agree.

## Prompt 8 — Optional visual models

First inspect hardware, dependency footprint, license, and runtime. Add an optional small detector or VLM only if it improves the staged demonstration. Do not download large weights or require paid APIs. Tie proposals to candidate/frame IDs, validate output, and keep manual annotation operational. Models cannot override failed localization or missing visibility evidence.

## Prompt 9 — Minimal user interface

Only after the CLI produces an end-to-end report, add a minimal local interface for two videos, processing status, evidence review, and report download. Do not require CUDA, large model weights, or a paid API.

## Prompt 10 — Evaluation and submission review

Audit against `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/TESTING.md`, and `docs/ROADMAP.md`. Run tests. Evaluate shared-map localization, object evidence, change claims, uncertainty, reproducibility, and failure cases. Report exact commands/results and distinguish automated outputs from manual annotations. Do not claim untested capability.
