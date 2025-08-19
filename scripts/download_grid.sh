#!/usr/bin/env bash
# How to use: bash download_grid.sh <path_to_your_dir>
# Example: bash download_grid.sh /mnt/DataNMVE/grid_dataset
set -euo pipefail

DEST="${1:-/mnt/DataNMVE/grid_dataset}"   # output folder (arg1)
PARALLEL="${2:-8}"                         # parallel downloads (arg2)
BASE="https://zenodo.org/records/3625687/files"

# --- tool checks ---
command -v wget  >/dev/null || { echo "Please install wget"; exit 1; }
command -v xargs >/dev/null || { echo "xargs not found"; exit 1; }
command -v unzip >/dev/null || { echo "Please install unzip"; exit 1; }

mkdir -p "$DEST"
cd "$DEST"

# --- files to fetch (speaker 21 has no video) ---
FILES=(audio_25k.zip alignments.zip)
for i in $(seq 1 34); do
  [[ $i -eq 21 ]] || FILES+=("s${i}.zip")
done

# --- download in parallel, resume if interrupted ---
printf '%s\n' "${FILES[@]}" | xargs -n1 -P "$PARALLEL" -I{} bash -lc '
  f="{}"
  url="'"$BASE"'/${f}?download=1"
  if [[ -f "$f" ]]; then
    echo "[skip] $f exists"
  else
    # -c resume, -nv quiet-ish, -O force proper filename (no ?download=1)
    wget -c -nv -O "$f" "$url"
  fi


# --- quick integrity check (unzip -tq) ---
echo "Verifying archives..."
for f in "${FILES[@]}"; do
  unzip -tq "$f" >/dev/null || { echo "CORRUPT: $f"; exit 2; }
done

# --- extract into tidy folders ---
echo "Extracting..."
mkdir -p audio_25k alignments
unzip -n audio_25k.zip -d audio_25k >/dev/null
unzip -n alignments.zip -d alignments >/dev/null
for f in s*.zip; do
  d="${f%.zip}"
  mkdir -p "$d"
  unzip -n "$f" -d "$d" >/dev/null
done

echo "Done → $DEST"
