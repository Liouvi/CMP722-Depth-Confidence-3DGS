#!/usr/bin/env bash
# Lay out LLFF for the baseline loader:
#   data/llff/<scene>/{images, sparse/0, poses_bounds.npy, <n>_views/dense/fused.ply}
#
# Inputs (download first, see README):
#   $1  nerf_llff_data.zip   images and COLMAP poses (NeRF real forward-facing data)
#   $2  nerf_llff_data.zip   from the DropGaussian authors: dense point clouds for 3/6/9 views
#
# The initial point clouds are the baseline authors' own, so the baseline is
# reproduced from the same initialisation they used.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <llff_images.zip> <dropgaussian_llff_pointclouds.zip>" >&2
  exit 2
fi
images_zip=$(realpath "$1")
pcd_zip=$(realpath "$2")
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/data/llff"
scenes=(fern flower fortress horns leaves orchids room trex)

tmp=$(mktemp -d "$root/data/.llff_extract.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
unzip -q "$images_zip" -d "$tmp/images"
unzip -q "$pcd_zip" -d "$tmp/pcd"

src=$(dirname "$(find "$tmp/images" -maxdepth 3 -type d -name fern | head -n 1)")
mkdir -p "$out"
for scene in "${scenes[@]}"; do
  [ -d "$src/$scene/images" ] && [ -d "$src/$scene/sparse/0" ] || { echo "missing data for $scene" >&2; exit 1; }
  mkdir -p "$out/$scene"
  [ -f "$src/$scene/poses_bounds.npy" ] || { echo "missing poses_bounds.npy for $scene" >&2; exit 1; }
  cp -r "$src/$scene/images" "$src/$scene/sparse" "$src/$scene/poses_bounds.npy" "$out/$scene/"
  for n in 3 6 9; do
    ply="$tmp/pcd/$scene/${n}_views/dense/fused.ply"
    [ -f "$ply" ] || { echo "missing point cloud $scene ${n}_views" >&2; exit 1; }
    mkdir -p "$out/$scene/${n}_views/dense"
    cp "$ply" "$out/$scene/${n}_views/dense/fused.ply"
  done
  echo "$scene: $(ls "$out/$scene/images" | wc -l) images"
done
