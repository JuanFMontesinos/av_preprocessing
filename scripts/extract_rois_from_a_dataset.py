"""
File: extract_rois_from_a_dataset.py
Author: Juan Montesinos
Created: 20/08/2025

Extract the ROIs to extract AV Hubert features from a directory of videos that has already been processed to have landmarks.

How to use:
    uv run scripts/extract_rois_from_a_dataset.py <video_suffix> [--dst_dir <destination_directory>]
Example:
    uv run scripts/extract_rois_from_a_dataset.py /mnt/DataNMVE/grid_dataset/ .mpg
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fire import Fire

import av_preprocessing as avp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def process_dataset(dataset_dir: str, video_suffix: str, dst_dir: Optional[str] = None):
    dataset_path = Path(dataset_dir)  # type: ignore
    if dst_dir is None:
        dst_path = dataset_path
    else:
        dst_path = Path(dst_dir)

    landmarks_path = dst_path / "landmarks"
    rois_path = dst_path / "rois"

    landmark_files = avp.paths.list_of_npy_files(landmarks_path)
    n_ld_files = len(landmark_files)

    logging.info(f"Found {n_ld_files} landmark files in {landmarks_path}")
    logging.info(f"ROIS for AV Hubert will be saved to {rois_path}")

    mean_face = np.load(avp.paths.ASSETS_PATH / "mean_face.npy")[:2].T.astype(np.float32)

    for i, landmark_file in enumerate(landmark_files):
        logging.info(f"[{i + 1}/{n_ld_files}] Processing {landmark_file}")
        landmarks = [l.T[:, :2] for l in np.load(landmark_file)]
        with landmark_file.with_suffix(".yaml").open("r") as f:
            metadata = yaml.safe_load(f)
        roi_output_path = rois_path / landmark_file.relative_to(landmarks_path).with_suffix(".mp4")
        roi_output_path.parent.mkdir(parents=True, exist_ok=True)
        rois = avp.functionals.crop_rois(
            input_video_path=(
                dataset_path / landmark_file.relative_to(landmarks_path).with_suffix(video_suffix)
            ).as_posix(),
            output_path=roi_output_path,
            landmarks=landmarks,
            mean_face=mean_face,
            fps=metadata.get("fps", 25),
        )


if __name__ == "__main__":
    # debug mode
    if len(sys.argv) < 2:
        process_dataset("/mnt/DataNMVE/grid_dataset", video_suffix=".mpg")
    else:
        Fire(process_dataset)
