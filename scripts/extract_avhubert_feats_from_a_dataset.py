# /// script
# requires-python = "~=3.9"
# dependencies = [
#   "numpy<1.20",
#   "scipy==1.10.0",
#   "opencv-python",
#   "torch<2.6",
#   "omegaconf==2.1.1",
#   "hydra-core==1.1.1",
#   "python-speech-features==0.6",
#   "scikit-image",
#    "tqdm",
#    "sentencepiece==0.1.96",
#   "fire",
# ]
# ///

"""
File: extract_avhubert_feats_from_a_dataset.py
Author: Juan Montesinos
Created: 20/08/2025

Extract AV Hubert features from a directory of videos that has already been processed to have landmarks and ROIs.
The features are stored as .npy files in a specified destination directory.
How to use:
    uv run --python 3.9 scripts/extract_avhubert_feats_from_a_dataset.py <video_suffix> [--dst_dir <destination_directory>] [--max_length <max_length>]
Example:
    uv run --python 3.9 scripts/extract_avhubert_feats_from_a_dataset.py /mnt/DataNMVE/grid_dataset .mp4
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fire import Fire

AVP_LIB_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(AVP_LIB_PATH))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import av_preprocessing as avp


def process_dataset(dataset_dir: str, video_suffix: str, dst_dir: Optional[str] = None, max_length: int = 250):
    dataset_path = Path(dataset_dir)  # type: ignore
    if dst_dir is None:
        dst_path = dataset_path
    else:
        dst_path = Path(dst_dir)

    landmarks_path = dst_path / "landmarks"
    av_feats_path = dst_path / "av_hubert_feats"
    rois_path = dst_path / "rois"
    if not rois_path.exists():
        logging.error(
            f"ROIs path {rois_path} does not exist. Please run uv run scripts/extract_rois_from_a_dataset.py <video_suffix> [--dst_dir <destination_directory>]"
        )
        sys.exit(1)
    if not landmarks_path.exists():
        logging.error(
            f"Landmarks path {landmarks_path} does not exist. Please run uv run scripts/extract_landmarks_from_a_dataset.py <dataset_dir> [--dst_dir <destination_dir>]"
        )
        sys.exit(1)
    av_feats_path.mkdir(parents=True, exist_ok=True)

    landmark_files = avp.paths.list_of_npy_files(landmarks_path)
    n_ld_files = len(landmark_files)

    logging.info(f"Found {n_ld_files} landmark files in {landmarks_path}")
    logging.info(f"AV Hubert features will be saved to {av_feats_path}")

    model, cfg, task = avp.av_hubert.instantiate_av_hubert_and_add_to_python_path()
    if torch.cuda.is_available():
        model = model.cuda()
    for i, landmark_file in enumerate(landmark_files):
        av_path = av_feats_path / landmark_file.relative_to(landmarks_path).with_suffix(".npy")
        roi_path = rois_path / landmark_file.relative_to(landmarks_path).with_suffix(".mp4")
        logging.info(f"[{i + 1}/{n_ld_files}] Processing {roi_path}")
        visual_feats = avp.functionals.extract_visual_feature(roi_path.as_posix(), model, task, max_length=max_length)
        visual_feats = visual_feats.cpu().numpy().astype(np.float16)
        av_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(av_path, visual_feats, allow_pickle=False)
        logging.info(f"\tSaved AV Hubert features to {av_path}")


if __name__ == "__main__":
    # debug mode
    if len(sys.argv) < 2:
        process_dataset("/mnt/DataNMVE/grid_dataset", video_suffix=".mp4")
    else:
        Fire(process_dataset)
