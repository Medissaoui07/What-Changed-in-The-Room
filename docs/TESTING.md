# Testing and Evaluation

## Test levels

Keep the baseline deterministic and runnable without COLMAP, a GPU, neural weights, network access, or paid APIs. Use synthetic point clouds and fake visual-model backends for unit tests. Optional real-COLMAP/model checks must be marked separately and skipped with a clear reason when unavailable.

## Synthetic registration cases

Create sparse point sets with known ground truth and test:

1. Identity alignment and an unchanged scene.
2. Known rotation and translation.
3. Known monocular scale difference and similarity-transform recovery.
4. Rigid-only alignment when metric scale is known.
5. Outliers and a changed cluster that should not dominate alignment.
6. Low overlap, repetitive/ambiguous geometry, and insufficient points.
7. Poor registration rejection and coordinate-frame/transform direction.

Report transformation error, inlier count/ratio, overlap, residual, and scale recovery. Set tolerances appropriate for each synthetic fixture and document them.

## Sparse change evidence cases

Test unchanged sparse scenes, candidate added/moved/removed clusters, partial image coverage, missing camera metadata, occlusion proxies, and registration failure. Verify that:

- a missing sparse point alone never proves removal;
- absent camera/frame evidence yields unknown or insufficient observation;
- failed alignment blocks confident change classification;
- outputs retain candidate/point IDs and linked frame evidence where available;
- semantic model output cannot override geometry or observation gates.

Sparse point clouds do not encode complete surface occupancy. Do not write tests that imply they do.

## Optional visual model tests

Use fake DINO/ViT, segmentation, or VLM backends to test input/output contracts, candidate/frame ID preservation, malformed responses, and model absence. Real model tests are optional, require deliberate weight installation, and should record model name/version, hardware, runtime, and license. Never download weights during ordinary test runs.

Measure whether a visual component improves object association or labeling on the same labeled examples used by the geometry baseline. Keep model-proposed labels separate from calibrated pipeline confidence.

## Small labeled evaluation set

Use recorded before/after examples with documented ground truth for:

- unchanged room;
- moved object;
- added object;
- removed object with clear visibility;
- partial/occluded observation;
- different camera paths and alignment failure.

Include sparse reconstruction outputs and source frame links. Record model failures and manual interventions.

## Metrics and reporting

Track:

- registration rotation, translation, and scale error where ground truth exists;
- registration inliers, overlap, and residual;
- candidate precision/recall at a declared spatial tolerance;
- supported change-type performance;
- false candidates on unchanged areas;
- abstention/uncertainty behavior;
- image-association and semantic-label correctness;
- runtime and peak memory.

Do not report a single accuracy number without a defined dataset, split, and metric. Sparse-only results must be described as candidate evidence, not complete surface change measurement.

## Manual review

- [ ] Both videos have enough static, textured overlap for sparse reconstruction.
- [ ] Scale mode is explicit.
- [ ] Transform direction is visible and tested.
- [ ] Stable room evidence supports alignment.
- [ ] Candidate findings link to source points and frames.
- [ ] Missing observations are marked unknown/insufficient.
- [ ] JSON and visualization agree.
- [ ] At least one ambiguous or failed case is shown honestly.
