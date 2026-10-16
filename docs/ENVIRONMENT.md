# Development Environment and Sparse COLMAP Workflow

## Environment notes

The original Phase 0 audit recorded Python 3.10.12, NumPy/SciPy/OpenCV/pytest/PyYAML, FFmpeg, no PyTorch/Open3D/Streamlit, and no working NVIDIA driver. That was a snapshot, not a guarantee of the current interpreter.

The user later installed COLMAP and produced sparse models. COLMAP is an optional geometry backend; do not rebuild it with CUDA or use the Colab dense-build notebook for the current path. CPU feature extraction, matching, and image registration are sufficient. If `colmap --version` fails because the Snap package cannot run in your environment, use a working distro COLMAP executable and pass its path with `--colmap /path/to/colmap`.

## Python setup

From `navigation_humanoid/`:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest
```

Install optional dependencies only when their roadmap phase needs them. Keep neural model frameworks and weights out of the baseline install. Avoid installing GUI OpenCV and headless OpenCV together in one environment.

## Export an existing sparse COLMAP model

If the GUI already produced a model in `project/sparse/0`, export its points as PLY:

```sh
colmap model_converter \
  --input_path project/sparse/0 \
  --output_path project/sparse.ply \
  --output_type PLY
```

Do this separately for before and after. Keep each original `sparse/` model and its images: PLY contains points/colors but loses camera poses and registered image names used to link evidence and reason about observation. If COLMAP produced multiple model folders, inspect them and choose the model with the useful registered images/points instead of assuming folder `0` is best.

## CPU sparse reconstruction from sampled images

Put extracted JPG/PNG frames into separate directories. For example:

```text
work/
  before/images/*.jpg
  after/images/*.jpg
```

Run the following for each scene (first `before`, then `after`). CPU feature extraction and matching are requested explicitly so this works with a non-CUDA COLMAP build:

```sh
SCENE=before
mkdir -p "work/$SCENE/sparse"
colmap feature_extractor \
  --database_path "work/$SCENE/database.db" \
  --image_path "work/$SCENE/images" \
  --SiftExtraction.use_gpu 0
colmap sequential_matcher \
  --database_path "work/$SCENE/database.db" \
  --SiftMatching.use_gpu 0
colmap mapper \
  --database_path "work/$SCENE/database.db" \
  --image_path "work/$SCENE/images" \
  --output_path "work/$SCENE/sparse"
colmap model_converter \
  --input_path "work/$SCENE/sparse/0" \
  --output_path "work/${SCENE}_sparse.ply" \
  --output_type PLY
```

Change `SCENE=before` to `SCENE=after` and repeat. Start with a modest set of well-spaced frames; more frames are not automatically better and CPU exhaustive matching can grow quadratically. Sequential matching is suitable for frames sampled in video order. If no `sparse/0` model is created, inspect mapper output and registered-image count before exporting.

The feature manifest and original frames should remain associated with their COLMAP image names. Separate reconstructions have separate coordinate frames and monocular scale is generally unknown; Phase 3 must estimate/record a similarity transform before comparison.

## Inspect the existing independent-map registration experiment

With the binary model folders and the databases created alongside each image set, run:

```sh
wcr register-colmap \
  output/colmap/before_model output/colmap/after_model \
  --before-database output/before/database.db \
  --after-database output/after/database.db
```

The command reports matched 3D pairs, RANSAC/ICP diagnostics, and the after-to-before transform as JSON. Exit code `3` means the available matches failed the registration quality gates; do not use a failed transform downstream. `--inlier-threshold` is expressed in before-model coordinate units and should be selected based on the reconstruction scale and noise, not assumed to be meters.

On the existing example, independent-map SIFT matching found 9 candidate 3D pairs but only 1 geometric inlier, so the command correctly rejects the transform. These files are not yet suitable for change claims.

## Localize after images into the before map

The preferred next experiment copies the before COLMAP database to a temporary workspace, extracts after-frame features, matches them against the reference images, and asks COLMAP to localize after cameras into the before model. It leaves the original database untouched. Run from the project root:

```sh
source .venv/bin/activate
python -m pip install -e .
wcr localize-colmap output/colmap/before_model \
  --before-database output/before/database.db \
  --after-images output/after \
  --output-dir output/shared_map
```

Pass either a model folder or the parent `sparse/` folder. For a parent folder, the command automatically chooses the submodel with the most registered images also listed in the database, breaking ties by point count. It prints the selected model, registered-camera count, and point count. Open the generated preview:

```sh
xdg-open output/shared_map/shared_map.html
```

The page shows a projected sparse map (gray points), before cameras (blue), after cameras localized into the before coordinate frame (red), and thumbnails of localized after frames. If zero after cameras register, the result is a failure/recapture signal, not evidence of no changes. The generated `shared_model/` contains the native COLMAP model. Keep the original input models and videos.

Temporary database/image files are staged under the requested output directory so Snap-confined COLMAP can access them; the temporary folder is removed when the command finishes.

At the current checked-in example paths, the binary PLY exports are named `before_sparse` and `after_sparse` without suffixes. They are valid PLY files, but the point-cloud importer requires `.ply`; rename them to `before_sparse.ply` and `after_sparse.ply` before using `wcr import-point-cloud`. Prefer the native model registration command above because it retains image-feature tracks.

## Optional visual models

DINO/ViT features, segmentation, and VLMs are optional. Before adding one, check available RAM/GPU, framework compatibility, model weight size/license, and inference latency. Run it on selected keyframes or candidate crops. Do not download weights without explicit approval, and do not make a paid API necessary for the project.
