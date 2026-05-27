#!/usr/bin/env bash
# Usage: ./scripts/run_chapter.sh ch06_tile [ARCH]
# Builds every example in the chapter and executes them in dependency order.
set -euo pipefail

CHAP="${1:-}"
ARCH="${2:-sm_80}"

if [[ -z "$CHAP" ]]; then
  echo "usage: $0 <chapter_dir> [arch]"
  echo "       chapters: $(ls code | tr '\n' ' ')"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/code/$CHAP"
[[ -d "$DIR" ]] || { echo "chapter not found: $DIR"; exit 1; }

echo "=== building $CHAP (ARCH=$ARCH) ==="
make -C "$DIR" ARCH="$ARCH" clean all

echo "=== running examples ==="
make -C "$DIR" ARCH="$ARCH" run
