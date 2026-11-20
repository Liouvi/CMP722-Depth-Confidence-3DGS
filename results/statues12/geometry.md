# statues12: depth at held-out views against statues-ref

Mean over all finished runs of each variant (scenes x seeds). Lower AbsRel is better; d1 and d05 are the fractions of pixels within 25% and 5% of the reference depth.

| Variant | Runs | AbsRel | d1 | d05 | Coverage |
|---|---|---|---|---|---|
| V0 | 17 | 0.3154 | 0.5260 | 0.2772 | 0.9817 |
| V1 | 17 | 0.2571 | 0.5624 | 0.2565 | 0.9967 |
| V2 | 17 | 0.2250 | 0.6354 | 0.3043 | 0.9979 |
| V3 | 17 | 0.2748 | 0.5855 | 0.3170 | 0.9810 |
| V4 | 17 | 0.2417 | 0.5879 | 0.2685 | 0.9962 |

Paired over common runs; p is a Wilcoxon test over per-scene mean differences.

| Comparison | Mean AbsRel diff | p (scenes) | Scenes improved |
|---|---|---|---|
| V1 − V0 | -0.0583 | 0.0007 | 14/17 |
| V2 − V0 | -0.0904 | 0.0000 | 17/17 |
| V2 − V1 | -0.0322 | 0.0032 | 15/17 |
| V3 − V1 | +0.0177 | 0.4586 | 7/17 |
| V4 − V1 | -0.0155 | 0.1324 | 11/17 |
| V2 − V4 | -0.0167 | 0.0887 | 12/17 |
