
# Project Instructions

## Mission

Build "What Changed in the Room?", an offline-first system that compares two handheld phone videos of the same indoor environment and identifies meaningful 3D changes.

The system must identify moved, added, and removed objects while accounting for camera viewpoint differences, reconstruction errors, occlusion, and incomplete observations.

## Engineering priorities

1. Correctness over visual impressiveness.
2. Explicit uncertainty over unsupported claims.
3. Object-level reasoning over raw point-cloud subtraction.
4. Reproducible tests over demonstrations that only work on one example.
5. A usable, evidence-grounded result over unnecessary model complexity.

## Required development behavior

- Read README.md and the relevant docs before implementing a feature.
- Implement one roadmap phase at a time.
- Inspect the existing code before changing it.
- Explain the proposed change briefly before implementation.
- Run relevant tests after each implementation.
- Report actual test outcomes, failures, and limitations.
- Never claim that an untested feature works.
- Do not silently replace working implementations with simplified stubs.
- Avoid unnecessary abstractions and dependencies.
- Do not download large model weights without explicit approval.
- Do not require paid APIs for the core pipeline.
- Never hard-code local paths, credentials, or API keys.

## Architecture rules

Separate the application into:
- video preprocessing;
- 3D reconstruction;
- coordinate-frame registration;
- visibility and observation reasoning;
- geometric change detection;
- object correspondence and movement detection;
- semantic interpretation;
- reporting and visualization.

Keep the reconstruction backend replaceable.

The geometry-based pipeline must work without neural semantic models.

## Scientific correctness

- Never treat a missing point as proof that an object was removed.
- Distinguish unobserved space from observed empty space.
- Validate registration before comparing geometry.
- Use stable room structure to estimate alignment.
- Track reconstruction quality and registration uncertainty.
- Do not assume monocular reconstruction provides metric scale.
- Distinguish object movement from unrelated object addition and removal when correspondence is supported.
- Do not invent semantic labels, coordinates, or confidence scores.
- If evidence is insufficient, return an explicit uncertain status.

## Change types

Support these result types:
- moved
- added
- removed
- possible_change
- insufficient_observation

An alignment failure must prevent confident downstream classification.

## Code quality

- Use Python type hints.
- Write small, testable functions.
- Add docstrings to public interfaces.
- Provide useful error messages.
- Keep configuration separate from algorithm logic.
- Use deterministic synthetic test fixtures where possible.
- Keep raw recordings, generated point clouds, model weights, and large outputs out of Git.

## Definition of done

A feature is complete only when:
1. The implementation exists.
2. Relevant tests pass.
3. Failure cases are handled.
4. Commands and outputs are documented.
5. Known limitations are stated honestly.
