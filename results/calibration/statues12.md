# Confidence calibration: statues12 (reference: statues-ref)

Per training view, the Depth Anything 3 depth is aligned to the reference depth by least-squares scale and shift; error is absolute relative error. AUSE: area between the confidence and oracle sparsification curves (0 is perfect). AURG: area between random removal and the confidence curve (> 0 means confidence ranks errors better than chance). Spearman: rank correlation of confidence with error (negative is the desired sign). Depth control: AURG (depth) removes pixels farthest-first by the prior's depth; partial Spearman is the confidence-error correlation with the depth rank partialled out; Spearman (c, d) is the rank correlation of confidence with depth.

| Scene | AUSE | AURG | AURG (depth) | Spearman | Partial Spearman | Spearman (c, d) | AbsRel of prior | Valid px |
|---|---|---|---|---|---|---|---|---|
| statue01 | 0.285 | 0.398 | 0.058 | -0.377 | -0.405 | -0.676 | 0.078 | 0.865 |
| statue02 | 0.262 | 0.383 | 0.075 | -0.405 | -0.470 | -0.695 | 0.079 | 0.801 |
| statue03 | 0.552 | -0.027 | -0.137 | 0.202 | -0.145 | -0.859 | 0.272 | 0.842 |
| statue04 | 0.248 | 0.427 | 0.188 | -0.431 | -0.375 | -0.861 | 0.203 | 0.898 |
| statue05 | 0.441 | 0.111 | -0.063 | -0.054 | -0.212 | -0.857 | 0.094 | 0.674 |
| statue06 | 0.311 | 0.348 | 0.201 | -0.348 | -0.310 | -0.889 | 0.101 | 0.615 |
| statue07 | 0.358 | 0.299 | 0.170 | -0.268 | -0.225 | -0.893 | 0.179 | 0.691 |
| statue08 | 0.358 | 0.309 | 0.112 | -0.312 | -0.313 | -0.873 | 0.138 | 0.840 |
| statue09 | 0.209 | 0.466 | 0.314 | -0.460 | -0.303 | -0.844 | 0.150 | 0.698 |
| statue10 | 0.258 | 0.379 | 0.090 | -0.385 | -0.440 | -0.758 | 0.141 | 0.633 |
| statue11 | 0.459 | 0.180 | 0.093 | -0.235 | -0.172 | -0.959 | 0.095 | 0.610 |
| statue12 | 0.622 | 0.023 | -0.269 | -0.011 | -0.263 | -0.900 | 0.105 | 0.626 |
| statue13 | 0.304 | 0.407 | 0.227 | -0.363 | -0.337 | -0.870 | 0.120 | 0.624 |
| statue14 | 0.368 | 0.268 | 0.150 | -0.158 | -0.245 | -0.924 | 0.148 | 0.603 |
| statue15 | 0.398 | 0.160 | -0.119 | -0.102 | -0.205 | -0.691 | 0.133 | 0.812 |
| statue16 | 0.486 | 0.191 | -0.180 | -0.200 | -0.360 | -0.830 | 0.100 | 0.795 |
| statue17 | 0.552 | 0.112 | -0.062 | -0.121 | -0.236 | -0.888 | 0.088 | 0.775 |
| **mean** | **0.381** | **0.261** | **0.050** | **-0.237** | **-0.295** | **-0.839** | **0.131** | **0.730** |

AURG > 0 in 16/17 scenes; AURG > AURG (depth) in 17/17.
