# Confidence-weighted depth supervision for sparse-view Gaussian Splatting

Course project, Advanced Computer Vision. The question: when a depth prior from
a foundation model supervises sparse-view 3D Gaussian Splatting, does weighting
the depth loss by the model's own confidence map help, and does that confidence
track real depth error?

## Status (2026-11-20): complete

All experiments are finished and the report is built (`report/report.pdf`).

- **LLFF, 3 views**: 8 scenes x 5 variants x seeds 0-2 (120 runs).
- **Hacettepe Statues, 12 views**: 17 self-captured 360-degree phone
  walk-arounds of campus sculptures (Samsung Galaxy S26 Ultra, FHD 60 fps, 1x,
  no HDR/flash/stabilisation), prepared by `scripts/prepare_statues.py`;
  17 statues x 5 variants, seed 0 (85 runs). Nothing was tuned on this set.
- Reference models on all non-test views (8 + 17), used by the calibration
  (`scripts/calibration.py`) and geometry (`cwdepth/calib/geometry.py`)
  analyses; a depth-weight sweep on the LLFF tuning scenes; fly-through
  videos in `demo/` (`cwdepth/flythrough.py`): `llff_horns.mp4` and
  `statue16.mp4` are tracked; fly-throughs of all 17 statues are rendered
  into `demo/statues/` (not tracked, about 300 MB) by
  `conda run -n acv-gs scripts/render_demos.sh`.

Main findings (25 scenes pooled, paired Wilcoxon over scenes):

- The plain Pearson depth loss (V1) beats DropGaussian in 23/25 scenes
  (+0.46 dB on LLFF, +0.90 dB on the statues) and makes held-out geometry
  more accurate.
- Confidence weighting (V2) does not beat its shuffled-confidence control
  (V4) in PSNR (+0.01 dB, p=0.79; worse in 8/8 LLFF scenes), but it gives the
  most accurate geometry (AbsRel vs. V4 -0.013, 17/25, p=0.039).
- DA3 confidence ranks depth errors better than chance and better than
  distance alone, but it is mostly a proxy for distance, so V2 removes
  supervision from the background. Hard masking by confidence (V3) loses
  most of the gain.

3D viewer: `conda run -n acv-gs python -m cwdepth.viewer`, then open
<http://127.0.0.1:8765>. It lists every trained model in `outputs/` (sparse-view
variants, reference models, the depth-weight sweep and the statue art models),
renders it in the browser (Gaussian splats, degree-0 colour) and can jump
between the capture cameras; switching model keeps the viewpoint.

Art models: `conda run -n acv-gs scripts/train_art.sh` trains each statue on
all 96 frames at full resolution for 30k iterations without DropGaussian's
dropping (`outputs/statues-art/`). They are for viewing only and not part of
the evaluation.

Results live in `results/`; `scripts/make_report_assets.py` turns them into
the report's tables, figures and quoted numbers, and
`cd report && tectonic report.tex` builds the PDF.

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
