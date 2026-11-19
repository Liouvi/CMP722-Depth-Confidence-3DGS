# llff3: depth at held-out views against llff-ref

Mean over all finished runs of each variant (scenes x seeds). Lower AbsRel is better; d1 and d05 are the fractions of pixels within 25% and 5% of the reference depth.

| Variant | Runs | AbsRel | d1 | d05 | Coverage |
|---|---|---|---|---|---|
| V0 | 24 | 0.1315 | 0.8389 | 0.4875 | 0.9922 |
| V1 | 24 | 0.1120 | 0.8744 | 0.5036 | 0.9981 |
| V2 | 24 | 0.1075 | 0.8775 | 0.5178 | 0.9982 |
| V3 | 24 | 0.1160 | 0.8518 | 0.4948 | 0.9921 |
| V4 | 24 | 0.1113 | 0.8749 | 0.5069 | 0.9984 |

Paired over common runs; p is a Wilcoxon test over per-scene mean differences.

| Comparison | Mean AbsRel diff | p (scenes) | Scenes improved |
|---|---|---|---|
| V1 − V0 | -0.0195 | 0.0391 | 7/8 |
| V2 − V0 | -0.0240 | 0.0156 | 7/8 |
| V2 − V1 | -0.0045 | 0.0547 | 7/8 |
| V3 − V1 | +0.0041 | 0.4609 | 3/8 |
| V4 − V1 | -0.0007 | 0.7422 | 4/8 |
| V2 − V4 | -0.0038 | 0.3828 | 5/8 |
