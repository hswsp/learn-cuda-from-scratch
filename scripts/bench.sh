#!/usr/bin/env bash
# Benchmark helpers: wraps Nsight Compute / Nsight Systems for selected binaries.
# Usage:
#   ./scripts/bench.sh ncu code/ch09_gemm/gemm_tiled         # detailed kernel metrics
#   ./scripts/bench.sh nsys code/ch08_async/multi_stream     # timeline
#   ./scripts/bench.sh time code/ch06_tile/tiled_matmul      # quick wall-clock
set -euo pipefail

TOOL="${1:-}"
BIN="${2:-}"

if [[ -z "$TOOL" || -z "$BIN" ]]; then
  echo "usage: $0 {ncu|nsys|time} <path/to/binary>"
  exit 1
fi

case "$TOOL" in
  ncu)
    # Full set takes a while; --section MemoryWorkloadAnalysis only is fast.
    ncu --set full -o "$(basename "$BIN").ncu-rep" "$BIN"
    echo "wrote $(basename "$BIN").ncu-rep — open in Nsight Compute UI"
    ;;
  nsys)
    nsys profile --stats=true --force-overwrite=true -o "$(basename "$BIN").nsys-rep" "$BIN"
    ;;
  time)
    /usr/bin/time -v "$BIN"
    ;;
  *)
    echo "unknown tool: $TOOL"
    exit 1
    ;;
esac
