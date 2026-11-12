#!/usr/bin/env bash
# Compute Depth Anything 3 depth and confidence for the training views of every
# LLFF scene. Run in the depth-model environment:
#
#   conda run -n acv-da3 scripts/compute_priors.sh da3-large depth-anything/DA3-LARGE-1.1 [n_views]
#   conda run -n acv-da3 scripts/compute_priors.sh da3-base  depth-anything/DA3-BASE      [n_views]
#
# One scene per process, each under a 3 GB RAM cap, so the GPU and host memory
# are released between scenes. Scenes that already have priors are skipped.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <name> <huggingface model id> [n_views=3]" >&2
  exit 2
fi
name="$1"
model="$2"
n_views="${3:-3}"
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

suffix=""
[ "$n_views" = "3" ] || suffix="-${n_views}v"
for scene in fern flower fortress horns leaves orchids room trex; do
  out="data/priors/${name}${suffix}/llff/$scene"
  if [ -f "$out/meta.json" ]; then
    echo "$scene: already done"
    continue
  fi
  scripts/capped.sh 3G python -m cwdepth.priors.da3 --scene_dir "data/llff/$scene" --n_views "$n_views" \
    --out_dir "$out" --model "$model" > "/tmp/cwdepth_prior_${name}_${scene}.log" 2>&1 \
    || { echo "$scene: FAILED, see /tmp/cwdepth_prior_${name}_${scene}.log" >&2; exit 1; }
  python - "$out/meta.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{m['scene']}: {len(m['views'])} views, {m['shape']}, peak {m['peak_vram_mb']:.0f} MB, "
      f"conf {m['conf_range'][0]:.2f}-{m['conf_range'][1]:.2f}")
EOF
done
