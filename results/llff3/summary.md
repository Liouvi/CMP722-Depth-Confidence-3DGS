# llff3

120 runs, 8 scenes. Mean and standard deviation are over seeds of the scene-averaged metric; a seed counts only if all scenes finished.

| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |
|---|---|---|---|---|---|---|
| V0 | 3 | 20.451 ± 0.078 | 0.709 ± 0.003 | 0.201 ± 0.002 | 76,672 | 2.1 |
| V1 | 3 | 20.914 ± 0.065 | 0.726 ± 0.001 | 0.189 ± 0.000 | 76,853 | 3.3 |
| V2 | 3 | 20.831 ± 0.109 | 0.723 ± 0.003 | 0.191 ± 0.001 | 77,387 | 3.0 |
| V3 | 3 | 20.576 ± 0.043 | 0.714 ± 0.001 | 0.199 ± 0.001 | 76,431 | 3.8 |
| V4 | 3 | 20.975 ± 0.052 | 0.726 ± 0.002 | 0.189 ± 0.001 | 77,011 | 3.9 |

## PSNR per scene (mean over seeds)

| Variant | fern | flower | fortress | horns | leaves | orchids | room | trex |
|---|---|---|---|---|---|---|---|---|
| V0 | 22.95 | 20.43 | 23.17 | 18.96 | 17.69 | 16.08 | 21.85 | 22.48 |
| V1 | 23.03 | 20.75 | 23.74 | 20.18 | 18.25 | 16.44 | 21.84 | 23.07 |
| V2 | 22.84 | 20.76 | 23.88 | 19.99 | 17.87 | 16.49 | 21.82 | 22.98 |
| V3 | 22.99 | 20.62 | 23.64 | 19.14 | 17.73 | 16.11 | 21.93 | 22.45 |
| V4 | 23.06 | 20.87 | 24.04 | 20.02 | 18.29 | 16.55 | 21.86 | 23.13 |

## Paired comparisons

Difference is first minus second. `p (runs)` is a Wilcoxon signed-rank test over scene × seed pairs; `p (scenes)` uses one seed-averaged difference per scene and is the conservative one, because seeds of the same scene are not independent.

| Comparison | Metric | Mean diff | p (runs) | p (scenes) | Scenes improved |
|---|---|---|---|---|---|
| V2 − V0 | PSNR | +0.3799 | 0.0004 | 0.0391 | 6/8 |
| V2 − V0 | SSIM | +0.0137 | 0.0000 | 0.0078 | 8/8 |
| V2 − V0 | LPIPS | -0.0100 | 0.0000 | 0.0078 | 8/8 |
| V2 − V1 | PSNR | -0.0826 | 0.2076 | 0.2500 | 3/8 |
| V2 − V1 | SSIM | -0.0028 | 0.0491 | 0.2500 | 2/8 |
| V2 − V1 | LPIPS | +0.0013 | 0.1280 | 0.3828 | 3/8 |
| V1 − V0 | PSNR | +0.4624 | 0.0000 | 0.0156 | 7/8 |
| V1 − V0 | SSIM | +0.0164 | 0.0000 | 0.0078 | 8/8 |
| V1 − V0 | LPIPS | -0.0114 | 0.0000 | 0.0078 | 8/8 |
| V2 − V4 | PSNR | -0.1442 | 0.0059 | 0.0078 | 0/8 |
| V2 − V4 | SSIM | -0.0035 | 0.0425 | 0.0781 | 2/8 |
| V2 − V4 | LPIPS | +0.0015 | 0.1780 | 0.2500 | 3/8 |
| V3 − V1 | PSNR | -0.3377 | 0.0002 | 0.0234 | 1/8 |
| V3 − V1 | SSIM | -0.0118 | 0.0001 | 0.0547 | 1/8 |
| V3 − V1 | LPIPS | +0.0094 | 0.0000 | 0.0078 | 0/8 |
| V4 − V1 | PSNR | +0.0616 | 0.0229 | 0.1484 | 7/8 |
| V4 − V1 | SSIM | +0.0007 | 0.7898 | 0.3828 | 6/8 |
| V4 − V1 | LPIPS | -0.0002 | 0.7469 | 0.7422 | 5/8 |
