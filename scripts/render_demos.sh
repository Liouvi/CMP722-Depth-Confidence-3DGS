#!/usr/bin/env bash
# Fly-through demo for every Hacettepe statue: V0 | V1 | V2 (seed 0) along a
# smoothed loop through all 96 camera poses, inverse rendered depth below.
# Videos go to demo/statues/ (not tracked by git; about 18 MB each).
# Existing videos are kept, so the script can be rerun after an interruption.
#
#   conda run -n acv-gs scripts/render_demos.sh [statue01 statue02 ...]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
out=demo/statues
mkdir -p "$out"
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  mapfile -t scenes < <(ls -d data/statues/statue*/ | xargs -n1 basename)
fi
for s in "${scenes[@]}"; do
  if [ -s "$out/$s.mp4" ]; then
    echo "$s: exists, skipping"
    continue
  fi
  runs=()
  for v in "V0 baseline" "V1 depth" "V2 depth+confidence"; do
    id=${v%% *}
    runs+=(--run "$v=outputs/statues12/$id/$s/seed0")
  done
  python -m cwdepth.flythrough --scene_dir "data/statues/$s" "${runs[@]}" \
    --out "$out/.$s.tmp.mp4" --resolution 2 --frames 300 --depth 2>&1 | tail -1
  mv "$out/.$s.tmp.mp4" "$out/$s.mp4"
done
