#!/usr/bin/env bash
# Build the baseline's two CUDA extensions into the active environment.
#
#   conda run -n acv-gs scripts/build_extensions.sh
#
# The submodule is left unmodified. Its sources rely on headers that older
# compilers pulled in implicitly (<cstdint>, <cfloat>); with GCC 13 these must
# be included explicitly, which is done here with forced-include flags instead
# of editing third-party files.
#
# Compilation runs one job at a time under a 3 GB memory cap: an unbounded
# parallel CUDA build can exhaust the RAM of a small machine.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
base="$root/third_party/DropGaussian/submodules"
forced="-include cstdint -include cfloat"
export NVCC_APPEND_FLAGS="$forced"
export CFLAGS="${CFLAGS:-} $forced"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export MAX_JOBS=1

for ext in diff-gaussian-rasterization simple-knn; do
  rm -rf "$base/$ext/build" "$base/$ext"/*.egg-info
  "$root/scripts/capped.sh" 3G pip install -q --no-build-isolation "$base/$ext"
  rm -rf "$base/$ext/build" "$base/$ext"/*.egg-info
  echo "built $ext"
done
