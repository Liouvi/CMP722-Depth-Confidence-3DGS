#!/usr/bin/env bash
# "Art" models of the Hacettepe statues: the cleanest reconstruction we can
# make, not part of the evaluation. Each statue is trained on all 96 frames
# (no held-out views) at full resolution (719x1283) with the standard 3DGS
# schedule: 30k iterations, densification until 15k at gradient threshold
# 0.0002, and DropGaussian's dropping switched off (it is a sparse-view
# regulariser). Initial points: COLMAP's sparse model of all 96 frames.
#
# Output: outputs/statues-art/<statue>/point_cloud/iteration_30000/point_cloud.ply
# Finished statues (run.json present) are skipped.
#
#   conda run -n acv-gs scripts/train_art.sh [statue01 ...]
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  mapfile -t scenes < <(ls -d data/statues/statue*/ | xargs -n1 basename)
fi
failed=()
for s in "${scenes[@]}"; do
  out=outputs/statues-art/$s
  if [ -f "$out/run.json" ]; then
    echo "$s: done, skipping"
    continue
  fi
  echo "=== $(date '+%F %T') $s"
  mkdir -p "$out"
  scripts/capped.sh 8G python -m cwdepth.train -s "data/statues/$s" -m "$out" --overwrite \
    --resolution 1 --n_views 0 --seed 0 --no_drop \
    --iterations 30000 --position_lr_max_steps 30000 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 > "$out/train.log" 2>&1
  grep -E "^done" "$out/train.log" || { tail -5 "$out/train.log"; failed+=("$s"); }
done
echo "=== $(date '+%F %T') finished; failed: ${failed[*]:-none}"
