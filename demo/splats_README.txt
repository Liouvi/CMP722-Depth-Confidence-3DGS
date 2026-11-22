Hacettepe Statues: 3D Gaussian Splatting models
================================================

One file per statue (statue01.splat ... statue17.splat). Each is a 3D
Gaussian Splatting reconstruction of the statue and its surroundings,
trained on all 96 frames of a phone video (Samsung S26 Ultra, 1080p) at full
resolution for 30,000 iterations.

How to open (no installation)
-----------------------------
1. Open https://superspl.at/editor in Chrome, Edge or Firefox.
2. Drag a .splat file onto the page.
3. Left-drag to orbit, right-drag to pan, scroll to zoom.

The models open upright and centred on the statue. Areas the camera
never saw (behind trees, far background) are blurry or missing; this is
expected.

Format: the common .splat layout (32 bytes per Gaussian). Colour is stored
without view-dependent effects, so surfaces look slightly flatter than in the
original training output.
