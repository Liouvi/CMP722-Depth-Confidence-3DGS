# llff3

40 runs, 8 scenes. Mean and standard deviation are over seeds of the scene-averaged metric; a seed counts only if all scenes finished.

| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |
|---|---|---|---|---|---|---|
| V0 | 3 | 20.451 ± 0.078 | 0.709 ± 0.003 | 0.201 ± 0.002 | 76,672 | 2.1 |
| V1 | 1 | 20.928 | 0.727 | 0.189 | 77,176 | 3.5 |
| V2 | 1 | 20.928 | 0.726 | 0.190 | 77,300 | 3.2 |

## PSNR per scene (mean over seeds)

| Variant | fern | flower | fortress | horns | leaves | orchids | room | trex |
|---|---|---|---|---|---|---|---|---|
| V0 | 22.95 | 20.43 | 23.17 | 18.96 | 17.69 | 16.08 | 21.85 | 22.48 |
| V1 | 23.08 | 20.83 | 23.55 | 20.30 | 18.18 | 16.54 | 21.78 | 23.18 |
| V2 | 22.97 | 21.05 | 24.07 | 20.09 | 17.92 | 16.47 | 22.01 | 22.85 |

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
| V1 − V0 | PSNR | +0.3870 | 0.0234 | 0.0234 | 7/8 |
| V1 − V0 | SSIM | +0.0166 | 0.0078 | 0.0078 | 8/8 |
| V1 − V0 | LPIPS | -0.0097 | 0.0078 | 0.0078 | 8/8 |
