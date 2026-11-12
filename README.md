# Confidence-weighted depth supervision for sparse-view Gaussian Splatting

Course project, Advanced Computer Vision. The question: when a depth prior from
a foundation model supervises sparse-view 3D Gaussian Splatting, does weighting
the depth loss by the model's own confidence map help, and does that confidence
track real depth error?

Status: work in progress. Nothing in this repository has been validated yet;
see `results/` once it exists.

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
