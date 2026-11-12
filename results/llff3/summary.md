# llff3

8 runs, 8 scenes. Mean and standard deviation are over seeds of the scene-averaged metric; a seed counts only if all scenes finished.

| Variant | Seeds | PSNR | SSIM | LPIPS | Gaussians | Train min |
|---|---|---|---|---|---|---|
| V0 | 1 | 20.541 | 0.710 | 0.198 | 76,342 | 1.9 |

## PSNR per scene (mean over seeds)

| Variant | fern | flower | fortress | horns | leaves | orchids | room | trex |
|---|---|---|---|---|---|---|---|---|
| V0 | 22.94 | 20.37 | 23.49 | 19.36 | 17.71 | 16.11 | 21.91 | 22.44 |

## Paired comparisons

Difference is first minus second. `p (runs)` is a Wilcoxon signed-rank test over scene × seed pairs; `p (scenes)` uses one seed-averaged difference per scene and is the conservative one, because seeds of the same scene are not independent.

| Comparison | Metric | Mean diff | p (runs) | p (scenes) | Scenes improved |
|---|---|---|---|---|---|
