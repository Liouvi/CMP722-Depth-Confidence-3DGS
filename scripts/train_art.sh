#!/usr/bin/env bash
# "Art" models of the Hacettepe statues: the cleanest reconstruction we can
# make, not part of the evaluation. Each statue is trained on all 96 frames
# (no held-out views) at full resolution (719x1283) with the standard 3DGS
# schedule: 30k iterations, densification until 15k, and DropGaussian's
# dropping switched off (it is a sparse-view regulariser). Initial points:
# COLMAP's sparse model of all 96 frames.
#
# Densification gradient threshold: 0.0005. The 3DGS default of 0.0002 grows
# foliage-heavy statues past 2M Gaussians and the 8 GB RAM cap within 4k
# iterations (statue02); statue01 was trained at 0.0002 before this change.
# A statue that still fails is retried once at 0.001. The threshold used is
# in each run's cfg_args.
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
  ok=0
  for threshold in 0.0005 0.001; do
    echo "=== $(date '+%F %T') $s (densify threshold $threshold)"
    mkdir -p "$out"
    scripts/capped.sh 8G python -m cwdepth.train -s "data/statues/$s" -m "$out" --overwrite \
      --resolution 1 --n_views 0 --seed 0 --no_drop \
      --iterations 30000 --position_lr_max_steps 30000 \
      --densify_until_iter 15000 --densify_grad_threshold "$threshold" > "$out/train.log" 2>&1
    if grep -E "^done" "$out/train.log"; then
      ok=1
      break
    fi
    tail -3 "$out/train.log"
  done
  [ "$ok" -eq 1 ] || failed+=("$s")
done
echo "=== $(date '+%F %T') finished; failed: ${failed[*]:-none}"
