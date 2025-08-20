#!/usr/bin/env bash
# How to use: bash download_and_preprocess_grid.sh <path_to_your_dir>
# Example: bash download_and_preprocess_grid.sh /mnt/DataNMVE/grid_dataset
set -euo pipefail
DEST="${1:-/mnt/DataNMVE/grid_dataset}"   # output folder (arg1)


# Install dependencies
pip install uv
echo "Installing 3DDFA"
sh scripts/setup_3ddfa.sh
echo "Installing AV-Hubert"
sh scripts/setup_avhubert.sh

# Download dataset
echo "Downloading GRID dataset to $DEST"
bash scripts/download_grid.sh "$DEST"
# Extract landmarks need to locate the face in the image
uv run scripts/extract_landmarks_from_a_dataset.py "$DEST"
# Extract the Region of Interest (ROI) from the video frames
uv run scripts/extract_rois_from_a_dataset.py "$DEST" .mpg
#Extract AV Hubert features from the ROIs
uv run --python 3.9 scripts/extract_avhubert_feats_from_a_dataset.py "$DEST" .mp4
# Resample audio to 16kHz
uv run scripts/resample_audio_dataset.py "$DEST"/audio_16kHz 16000
