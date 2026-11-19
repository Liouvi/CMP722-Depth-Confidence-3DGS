# Confidence calibration: llff3 (reference: llff-ref)

Per training view, the Depth Anything 3 depth is aligned to the reference depth by least-squares scale and shift; error is absolute relative error. AUSE: area between the confidence and oracle sparsification curves (0 is perfect). AURG: area between random removal and the confidence curve (> 0 means confidence ranks errors better than chance). Spearman: rank correlation of confidence with error (negative is the desired sign).

| Scene | AUSE | AURG | Spearman | AbsRel of prior | Valid px |
|---|---|---|---|---|---|
| fern | 0.360 | 0.284 | -0.262 | 0.069 | 0.920 |
| flower | 0.438 | 0.121 | -0.090 | 0.095 | 0.675 |
| fortress | 0.403 | 0.203 | -0.420 | 0.101 | 0.974 |
| horns | 0.254 | 0.367 | -0.489 | 0.140 | 0.967 |
| leaves | 0.460 | 0.151 | -0.064 | 0.190 | 0.453 |
| orchids | 0.223 | 0.463 | -0.467 | 0.076 | 0.663 |
| room | 0.436 | 0.210 | -0.226 | 0.198 | 0.987 |
| trex | 0.609 | 0.030 | -0.081 | 0.082 | 0.997 |
| **mean** | **0.398** | **0.229** | **-0.262** | **0.119** | **0.829** |

AURG > 0 in 8/8 scenes.
