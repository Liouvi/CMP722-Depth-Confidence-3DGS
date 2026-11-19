#!/usr/bin/env bash
# Every GPU job after the LLFF 3-view matrix, one at a time (two training jobs
# on the 8 GB laptop GPU run at half speed each, so nothing is gained by
# running them together). Each step is resumable; rerunning the script skips
# finished work.
#
#   conda run -n acv-gs scripts/run_queue.sh
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
gs=(conda run --no-capture-output -n acv-gs)
da3=(conda run --no-capture-output -n acv-da3)
step() { echo "=== $(date '+%F %T') $*"; }

wait_for() {  # wait until no process matches the pattern
  while pgrep -f "$1" > /dev/null; do sleep 60; done
}

step "waiting for the LLFF 3-view matrix and statue preparation"
wait_for "run_matrix.py configs/llff3.yaml"
step "LLFF 3-view matrix (resume)"
"${gs[@]}" scripts/capped.sh 3G python scripts/run_matrix.py configs/llff3.yaml
step "LLFF reference models"
"${gs[@]}" scripts/capped.sh 5G python scripts/run_matrix.py configs/llff-ref.yaml
wait_for "prepare_statues.py"
step "statue priors"
"${da3[@]}" scripts/compute_priors.sh da3-large depth-anything/DA3-LARGE-1.1 12 statues
step "statues, seed 0, all variants"
"${gs[@]}" scripts/capped.sh 3G python scripts/run_matrix.py configs/statues12.yaml --seeds 0
step "statue reference models"
"${gs[@]}" scripts/capped.sh 5G python scripts/run_matrix.py configs/statues-ref.yaml
step "depth-weight sensitivity on the LLFF tuning scenes"
for w in 0.01 0.02 0.1 0.2; do
  "${gs[@]}" scripts/capped.sh 3G python scripts/run_matrix.py configs/llff3.yaml --scenes fern horns \
    --seeds 0 --variants V1 V2 --tag "w$w" --set "depth_weight=$w"
done
step "done"
