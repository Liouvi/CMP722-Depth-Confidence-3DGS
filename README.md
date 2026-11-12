# Confidence-weighted depth supervision for sparse-view Gaussian Splatting

Course project, Advanced Computer Vision. The question: when a depth prior from
a foundation model supervises sparse-view 3D Gaussian Splatting, does weighting
the depth loss by the model's own confidence map help, and does that confidence
track real depth error?

## Status (2026-11-12)

Proposal sent. The pipeline runs end to end on an RTX 5060 Laptop (8 GB).

- Baseline reproduced on LLFF 3-view over three seeds: 20.45 ± 0.08 PSNR,
  0.709 SSIM, 0.201 LPIPS (published: 20.76 / 0.713 / 0.200).
- Single-seed pilot, depth weight 0.05 (untuned): the plain depth loss (V1)
  reaches 20.93 PSNR against 20.54 for the baseline with the same seed; the
  confidence-weighted loss (V2) is indistinguishable from V1.
- See `results/llff3/summary.md` for the current table.

Still to do: control variants V3 and V4, seeds 1 and 2 for all depth variants,
depth-weight sensitivity, reference models and the calibration analysis,
Mip-NeRF 360 (12 views), figures, demo and report.

## Running on a shared machine

Every build and training command goes through `scripts/capped.sh`, which puts
a hard RAM cap on the job, and extensions are built with
`scripts/build_extensions.sh` (one compiler job at a time). An unbounded
parallel CUDA build froze the development machine once.

    conda run -n acv-gs scripts/build_extensions.sh
    conda run -n acv-gs scripts/capped.sh 3G python scripts/run_matrix.py configs/llff3.yaml
    conda run -n acv-gs python scripts/aggregate.py llff3

`run_matrix.py` skips finished runs, so it can be stopped and restarted.
Reference models (`configs/llff-ref.yaml`) load every training photo at full
resolution and need a 5 GB cap.

## Layout

| Path | Contents |
|---|---|
| `third_party/DropGaussian/` | Baseline (CVPR 2025), unmodified git submodule |
| `cwdepth/losses/depth.py` | Weighted Pearson depth loss and weight construction |
| `cwdepth/render.py` | Colour pass of the baseline plus a depth pass |
| `cwdepth/train.py` | Training entry point; `--depth_weight 0` is the baseline |
| `cwdepth/priors/` | Depth Anything 3 inference and the cache format |
| `configs/` | One file per benchmark: scenes, seeds, variants |
| `scripts/run_matrix.py` | Runs scene × variant × seed, resumable |
| `scripts/aggregate.py` | Builds `results/<benchmark>/runs.csv` and `summary.md` |
| `tests/` | Unit tests |
| `proposal/`, `report/` | LaTeX sources |

`data/` and `outputs/` are not tracked.

## Variants

| ID | Depth loss |
|---|---|
| V0 | none (DropGaussian baseline) |
| V1 | Pearson, uniform weights |
| V2 | Pearson, weighted by rank-normalised confidence (proposed) |
| V3 | Pearson on the most confident half of the pixels |
| V4 | Pearson, confidence weights placed at random (control) |

## Licences

The baseline and its rasteriser derive from the Inria 3D Gaussian Splatting
code, which is licensed for non-commercial research and evaluation use only
(`third_party/DropGaussian/LICENSE_GAUSSIAN_SPLATTING.md`). This project is
coursework and is bound by that licence. Depth Anything 3 Large is released
under CC BY-NC 4.0.
