# llff3

24 runs, 8 scenes. Mean and standard deviation are over seeds of the scene-averaged metric; a seed counts only if all scenes finished.

| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |
|---|---|---|---|---|---|---|
| V0 | 3 | 20.451 ± 0.078 | 0.709 ± 0.003 | 0.201 ± 0.002 | 76,672 | 2.1 |

## PSNR per scene (mean over seeds)

| Variant | fern | flower | fortress | horns | leaves | orchids | room | trex |
|---|---|---|---|---|---|---|---|---|
| V0 | 22.95 | 20.43 | 23.17 | 18.96 | 17.69 | 16.08 | 21.85 | 22.48 |

## Paired comparisons

Difference is first minus second. `p (runs)` is a Wilcoxon signed-rank test over scene × seed pairs; `p (scenes)` uses one seed-averaged difference per scene and is the conservative one, because seeds of the same scene are not independent.

| Comparison | Metric | Mean diff | p (runs) | p (scenes) | Scenes improved |
|---|---|---|---|---|---|
