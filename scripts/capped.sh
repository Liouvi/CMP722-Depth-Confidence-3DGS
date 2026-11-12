#!/usr/bin/env bash
# Run a command under a hard RAM cap, without swap, at low CPU and IO priority.
#
#   scripts/capped.sh 3G python -m cwdepth.train ...
#
# If the command exceeds the cap, the kernel kills that command only; the rest
# of the machine (for example another training job) is not affected. Use this
# for every build and training run on a shared machine.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <memory cap, e.g. 3G> <command> [args...]" >&2
  exit 2
fi
cap="$1"
shift

# Keep CPU thread pools small so a co-running job is not starved.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# One compiler process at a time for CUDA extension builds (torch honours MAX_JOBS).
export MAX_JOBS="${MAX_JOBS:-1}"

exec systemd-run --user --scope --quiet --collect \
  -p MemoryMax="$cap" -p MemorySwapMax=0 \
  nice -n 10 ionice -c 3 "$@"
