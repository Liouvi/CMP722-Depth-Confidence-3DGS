# llff3

57 runs, 8 scenes. Mean and standard deviation are over seeds of the scene-averaged metric; a seed counts only if all scenes finished.

| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |
|---|---|---|---|---|---|---|
| V0 | 3 | 20.451 ± 0.078 | 0.709 ± 0.003 | 0.201 ± 0.002 | 76,672 | 2.1 |
| V1 | 1 | 20.928 | 0.727 | 0.189 | 76,816 | 3.8 |
| V2 | 1 | 20.928 | 0.726 | 0.190 | 77,300 | 3.2 |
| V3 | 1 | 20.580 | 0.712 | 0.200 | 76,492 | 5.7 |
| V4 | 1 | 20.915 | 0.725 | 0.190 | 77,054 | 5.9 |

## PSNR per scene (mean over seeds)

| Variant | fern | flower | fortress | horns | leaves | orchids | room | trex |
|---|---|---|---|---|---|---|---|---|
| V0 | 22.95 | 20.43 | 23.17 | 18.96 | 17.69 | 16.08 | 21.85 | 22.48 |
| V1 | 22.97 | 20.83 | 23.55 | 20.30 | 18.18 | 16.54 | 21.78 | 23.18 |
| V2 | 22.97 | 21.05 | 24.07 | 20.09 | 17.92 | 16.47 | 22.01 | 22.85 |
| V3 | 23.00 | 20.67 | 23.25 | 19.13 | 17.73 | 16.07 | 22.04 | 22.74 |
| V4 | 23.10 | 20.99 | 24.03 | 19.68 | 18.27 | 16.50 | 21.71 | 23.04 |

## Paired comparisons

Difference is first minus second. `p (runs)` is a Wilcoxon signed-rank test over scene × seed pairs; `p (scenes)` uses one seed-averaged difference per scene and is the conservative one, because seeds of the same scene are not independent.

| Comparison | Metric | Mean diff | p (runs) | p (scenes) | Scenes improved |
|---|---|---|---|---|---|
| V2 − V0 | PSNR | +0.3867 | 0.0078 | 0.0078 | 8/8 |
| V2 − V0 | SSIM | +0.0153 | 0.0078 | 0.0078 | 8/8 |
| V2 − V0 | LPIPS | -0.0086 | 0.0078 | 0.0078 | 8/8 |
| V2 − V1 | PSNR | -0.0003 | 0.9453 | 0.9453 | 3/8 |
| V2 − V1 | SSIM | -0.0013 | 0.5469 | 0.5469 | 3/8 |
| V2 − V1 | LPIPS | +0.0010 | 0.6406 | 0.6406 | 4/8 |
| V1 − V0 | PSNR | +0.3329 | 0.0391 | 0.0391 | 7/8 |
| V1 − V0 | SSIM | +0.0155 | 0.0039 | 0.0078 | 8/8 |
| V1 − V0 | LPIPS | -0.0094 | 0.0039 | 0.0078 | 8/8 |
| V2 − V4 | PSNR | +0.0129 | 0.9453 | 0.9453 | 4/8 |
| V2 − V4 | SSIM | +0.0010 | 1.0000 | 1.0000 | 3/8 |
| V2 − V4 | LPIPS | -0.0001 | 0.7422 | 0.7422 | 6/8 |
| V3 − V1 | PSNR | -0.3485 | 0.0391 | 0.0391 | 1/8 |
| V3 − V1 | SSIM | -0.0149 | 0.0078 | 0.0078 | 0/8 |
| V3 − V1 | LPIPS | +0.0109 | 0.0156 | 0.0156 | 1/8 |
| V4 − V1 | PSNR | -0.0132 | 1.0000 | 1.0000 | 4/8 |
| V4 − V1 | SSIM | -0.0024 | 0.2500 | 0.2500 | 2/8 |
| V4 − V1 | LPIPS | +0.0011 | 0.8438 | 0.8438 | 3/8 |
