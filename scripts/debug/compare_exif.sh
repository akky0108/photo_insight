#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: bash scripts/debug/compare_exif.sh <file1> <file2>"
  exit 2
fi

f1="$1"
f2="$2"

tmp1="$(mktemp)"
tmp2="$(mktemp)"

trap 'rm -f "$tmp1" "$tmp2"' EXIT

echo "[INFO] file1=$f1"
echo "[INFO] file2=$f2"

exiftool "$f1" | sort > "$tmp1"
exiftool "$f2" | sort > "$tmp2"

diff -u "$tmp1" "$tmp2" || true