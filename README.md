# Confidence-weighted depth supervision for sparse-view Gaussian Splatting

Course project, Advanced Computer Vision. The question: when a depth prior from
a foundation model supervises sparse-view 3D Gaussian Splatting, does weighting
the depth loss by the model's own confidence map help, and does that confidence
track real depth error?

## Status (2026-11-19)

Two benchmarks:

- **LLFF, 3 views** (8 scenes, seeds 0-2, variants V0-V4).
- **Hacettepe Statues, 12 views**: 17 self-captured 360-degree phone
  walk-arounds of campus sculptures (Samsung Galaxy S26 Ultra, FHD 60 fps, 1x,
  no HDR/flash/stabilisation), prepared by `scripts/prepare_statues.py`,
  seed 0, variants V0-V4. Nothing was tuned on this set.

Plus reference models trained on all non-test views for the calibration
(`scripts/calibration.py`) and geometry (`cwdepth/calib/geometry.py`)
analyses, a depth-weight sweep on the LLFF tuning scenes, fly-through videos
(`cwdepth/flythrough.py`) and the report (`report/`). Results live in
`results/`; `scripts/make_report_assets.py` turns them into the report's
tables, figures and quoted numbers.

## Running on a shared machine

Every build and training command goes through `scripts/capped.sh`, which puts
a hard RAM cap on the job, and extensions are built with
`scripts/build_extensions.sh` (one compiler job at a time). An unbounded
parallel CUDA build froze the development machine once.

    conda run -n acv-gs scripts/build_extensions.sh
    conda run -n acv-gs scripts/capped.sh 3G python scripts/run_matrix.py configs/llff3.yaml
    conda run -n acv-gs python scripts/aggregate.py llff3

`run_matrix.py` skips finished runs, so it can be stopped and restarted.
`scripts/run_queue.sh` runs every GPU job after the LLFF matrix in order (one
job at a time: two jobs on the 8 GB GPU each run at half speed).

Statues, from the raw videos in `data/Hacettepe Statues/`:

    conda run -n acv-da3 scripts/capped.sh 4G python scripts/prepare_statues.py \
        --videos "data/Hacettepe Statues" --out data/statues
    conda run -n acv-da3 scripts/compute_priors.sh da3-large depth-anything/DA3-LARGE-1.1 12 statues
    conda run -n acv-gs scripts/capped.sh 3G python scripts/run_matrix.py configs/statues12.yaml --seeds 0

Analysis and report:

    conda run -n acv-gs python -m cwdepth.calib.geometry configs/llff3.yaml --reference llff-ref
    conda run -n acv-gs python scripts/calibration.py configs/llff3.yaml --reference llff-ref
    conda run -n acv-gs python scripts/make_report_assets.py
    cd report && conda run -n latex tectonic report.tex
Reference models (`configs/llff-ref.yaml`) load every training photo at full
resolution and need a 5 GB cap.

## Layout

| Path | Contents |
|---|---|
| `third_party/DropGaussian/` | Baseline (CVPR 2025), unmodified git submodule |
| `cwdepth/losses/depth.py` | Weighted Pearson depth loss and weight construction |
| `cwdepth/render.py` | Colour pass of the baseline plus a depth pass |
| `cwdepth/train.py` | Training entry point; `--depth_weight 0` is the baseline |
| `cwdepth/calib/` | Confidence calibration (sparsification) and test-view depth accuracy |
| `cwdepth/flythrough.py` | Side-by-side fly-through videos along the capture path |
| `scripts/prepare_statues.py` | Videos to COLMAP scenes: frame sampling, SfM, 12-view initial cloud |
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
