# statues12

85 runs, 17 scenes. Mean and standard deviation are over seeds of the scene-averaged metric; a seed counts only if all scenes finished.

| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |
|---|---|---|---|---|---|---|
| V0 | 1 | 17.093 | 0.438 | 0.477 | 176,189 | 2.2 |
| V1 | 1 | 17.990 | 0.449 | 0.456 | 203,486 | 3.2 |
| V2 | 1 | 18.206 | 0.457 | 0.449 | 216,350 | 3.7 |
| V3 | 1 | 17.154 | 0.443 | 0.473 | 174,874 | 3.0 |
| V4 | 1 | 18.130 | 0.452 | 0.452 | 208,891 | 3.2 |

## PSNR per scene (mean over seeds)

| Variant | statue01 | statue02 | statue03 | statue04 | statue05 | statue06 | statue07 | statue08 | statue09 | statue10 | statue11 | statue12 | statue13 | statue14 | statue15 | statue16 | statue17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 18.39 | 18.35 | 16.76 | 19.01 | 15.80 | 17.86 | 16.38 | 16.35 | 17.43 | 16.56 | 17.67 | 17.79 | 15.76 | 16.80 | 16.20 | 16.08 | 17.39 |
| V1 | 19.94 | 19.77 | 18.06 | 19.43 | 16.34 | 18.39 | 17.47 | 17.31 | 17.79 | 17.57 | 19.44 | 17.97 | 16.24 | 16.74 | 17.19 | 18.08 | 18.11 |
| V2 | 20.41 | 19.36 | 18.23 | 19.72 | 16.62 | 18.78 | 17.35 | 17.80 | 18.90 | 17.44 | 19.44 | 18.43 | 16.46 | 17.12 | 16.94 | 18.60 | 17.91 |
| V3 | 18.35 | 18.65 | 17.15 | 19.45 | 16.07 | 17.53 | 16.53 | 16.65 | 17.96 | 15.42 | 18.66 | 17.96 | 15.74 | 16.48 | 15.44 | 16.02 | 17.54 |
| V4 | 20.13 | 19.04 | 17.88 | 19.60 | 17.20 | 18.46 | 17.67 | 18.02 | 18.68 | 17.28 | 18.98 | 18.57 | 16.27 | 16.95 | 17.04 | 18.55 | 17.88 |

## Paired comparisons

Difference is first minus second. `p (runs)` is a Wilcoxon signed-rank test over scene × seed pairs; `p (scenes)` uses one seed-averaged difference per scene and is the conservative one, because seeds of the same scene are not independent.

| Comparison | Metric | Mean diff | p (runs) | p (scenes) | Scenes improved |
|---|---|---|---|---|---|
| V2 − V0 | PSNR | +1.1126 | 0.0000 | 0.0000 | 17/17 |
| V2 − V0 | SSIM | +0.0195 | 0.0000 | 0.0000 | 17/17 |
| V2 − V0 | LPIPS | -0.0282 | 0.0000 | 0.0000 | 17/17 |
| V2 − V1 | PSNR | +0.2155 | 0.0267 | 0.0267 | 11/17 |
| V2 − V1 | SSIM | +0.0085 | 0.0007 | 0.0007 | 14/17 |
| V2 − V1 | LPIPS | -0.0070 | 0.0267 | 0.0267 | 13/17 |
| V1 − V0 | PSNR | +0.8971 | 0.0000 | 0.0000 | 16/17 |
| V1 − V0 | SSIM | +0.0111 | 0.0001 | 0.0001 | 16/17 |
| V1 − V0 | LPIPS | -0.0211 | 0.0013 | 0.0013 | 15/17 |
| V2 − V4 | PSNR | +0.0760 | 0.1743 | 0.1743 | 12/17 |
| V2 − V4 | SSIM | +0.0052 | 0.0021 | 0.0021 | 14/17 |
| V2 − V4 | LPIPS | -0.0038 | 0.1089 | 0.1089 | 12/17 |
| V3 − V1 | PSNR | -0.8365 | 0.0002 | 0.0002 | 2/17 |
| V3 − V1 | SSIM | -0.0057 | 0.2435 | 0.2435 | 8/17 |
| V3 − V1 | LPIPS | +0.0176 | 0.0448 | 0.0448 | 6/17 |
| V4 − V1 | PSNR | +0.1395 | 0.2842 | 0.2842 | 11/17 |
| V4 − V1 | SSIM | +0.0033 | 0.1901 | 0.1901 | 9/17 |
| V4 − V1 | LPIPS | -0.0033 | 0.4874 | 0.4874 | 9/17 |
