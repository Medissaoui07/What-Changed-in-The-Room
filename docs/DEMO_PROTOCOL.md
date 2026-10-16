# Room-Change Demo Recording Protocol

## Goal and limitation

The demo shows whether a system can compare the same room across two phone recordings, put the evidence in shared 3D coordinates, and explain visible candidate changes. COLMAP is one possible mapping tool, not the goal. Sparse points are image features rather than complete surfaces. Present results as spatial candidates and show uncertainty rather than claiming a complete 3D inventory.

## Choose and prepare a scene

- Use one small room or desk area with stable textured structure and several distinguishable objects.
- Keep walls, large furniture, and other alignment anchors unchanged.
- Avoid mirrors, glass-heavy views, reflective surfaces, low light, and moving people.
- Include visible texture and depth variation; featureless walls and rotation-only camera motion give weak sparse geometry.
- Record the intended changes and their approximate locations in a ground-truth note before running the system.

## Record the before video

- Record about 30–60 seconds, moving slowly with little motion blur.
- Walk around the scene with translational movement and broad overlap between views; avoid only rotating in place.
- Show stable room structure and each target object from more than one angle.
- Keep the original video and frame manifest.

## Make and document changes

Use a small number of clear changes:

1. Move a chair or box to a different visible position.
2. Add a distinct object.
3. Optionally remove an object, ensuring its former location is clearly filmed afterward.
4. Keep the main room structure unchanged for alignment.

Write down the object, change type, before/after position, and whether each location is visible in both videos.

## Record the after video

- Capture the same room with a different but overlapping camera path.
- Revisit stable structural areas so after-video frames can localize into the before reference map.
- Clearly film the changed object and its old/new locations.
- Include a deliberate partial-visibility case only when demonstrating `insufficient_observation`.

## Process and review

Save the native map/localization outputs, camera poses, images, frame manifests, and ground-truth note together. If COLMAP is used, preserve its native sparse model and database; PLY alone loses camera poses and image tracks.

Review which after frames localized, shared-map coverage, candidate labels/anchors, and linked frames. Compare findings to ground truth and record false positives, misses, and uncertain outcomes. Mark manually supplied labels. Do not edit results without documenting it.

## Presentation

Show:

- the source videos and keyframes;
- the shared map and both camera paths, labeled as sparse if applicable;
- localization/coordinate evidence;
- change candidates with linked before/after evidence;
- uncertainty/insufficient-observation behavior;
- at least one failure or ambiguous case.

Do not present sparse points as a dense room surface or claim reliable removals from missing points alone.
