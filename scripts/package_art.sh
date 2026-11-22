#!/usr/bin/env bash
# Export every finished art model as an upright, centred .splat file and zip
# them with a how-to-open note, for sharing:
#
#   demo/splats/statueNN.splat
#   demo/hacettepe_statues_splats.zip   (the .splat files + README.txt)
#
#   conda run -n acv-gs scripts/package_art.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
python -m cwdepth.export_splat --out demo/splats
zip_path=demo/hacettepe_statues_splats.zip
tmp=demo/.hacettepe_statues_splats.tmp.zip
rm -f "$tmp"
zip -j -q "$tmp" demo/splats/statue*.splat
# Store the note as README.txt inside the archive.
(cd demo && cp splats_README.txt README.txt && zip -q "../$tmp" README.txt && rm README.txt)
mv "$tmp" "$zip_path"
ls -l "$zip_path"
unzip -l "$zip_path" | tail -3
